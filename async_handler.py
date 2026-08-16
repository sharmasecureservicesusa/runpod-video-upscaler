import asyncio
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import boto3
import requests
import runpod

# Thread executor to handle blocking CPU/GPU tasks without freezing asyncio
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

print("Available Environment Variable Keys:", list(os.environ.keys()), flush=True)


def download_video(job_input: dict, local_destination: str):
    """
    Downloads video chunks automatically:
    - If URL is Signed (contains '?'), downloads via HTTP requests.get()
    - If URL is Unsigned (no '?') or an s3:// URI, downloads via boto3 S3 client
    """
    video_url = job_input.get("video_url", "")
    if not video_url:
        raise ValueError("Missing 'video_url' in job_input.")

    s3_bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")
    
    # Check if URL contains query parameters ('?' indicates a presigned URL)
    is_signed = "?" in video_url
    is_s3_uri = video_url.startswith("s3://")

    # Use boto3 if it's an unsigned S3 URL or an s3:// URI
    if (is_s3_uri or not is_signed) and (s3_bucket or is_s3_uri):
        try:
            if is_s3_uri:
                parts = video_url.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                s3_key = parts[1] if len(parts) > 1 else ""
            else:
                bucket = s3_bucket
                # Extract S3 Key from URL
                if f"{bucket}/" in video_url:
                    s3_key = video_url.split(f"{bucket}/")[-1]
                else:
                    s3_key = video_url.split("/")[-1]

            key_id = job_input.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = job_input.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
            region = job_input.get("aws_region") or os.getenv("AWS_REGION", "eu-north1")
            endpoint_url = job_input.get("s3_endpoint_url") or os.getenv("S3_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud:443")

            s3_client = boto3.client(
                "s3",
                aws_access_key_id=key_id,
                aws_secret_access_key=secret_key,
                region_name=region,
                endpoint_url=endpoint_url
            )
            print(f"Downloading unsigned object via boto3: bucket='{bucket}', key='{s3_key}'", flush=True)
            s3_client.download_file(bucket, s3_key, local_destination)

        except Exception as e:
            # Fallback to HTTP request if boto3 fails and URL starts with http
            if video_url.startswith("http"):
                print(f"boto3 download failed ({e}), attempting direct HTTP request...", flush=True)
                res = requests.get(video_url, stream=True)
                res.raise_for_status()
                with open(local_destination, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                raise e
    else:
        # Download presigned / public HTTP URL
        print(f"Downloading signed/public URL via HTTP requests: {video_url[:60]}...", flush=True)
        res = requests.get(video_url, stream=True)
        res.raise_for_status()
        with open(local_destination, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

    # Validate that downloaded file exists and is non-empty
    if not os.path.exists(local_destination) or os.path.getsize(local_destination) == 0:
        raise ValueError(f"Downloaded video file is empty (0 bytes) from URL: {video_url}")


def get_video_info(input_video_path: str):
    """Probes video details safely using ffprobe with captured error logging."""
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found at: {input_video_path}")

    if os.path.getsize(input_video_path) == 0:
        raise ValueError(f"Input video file is empty (0 bytes) at: {input_video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(input_video_path)
    ]

    try:
        res = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8")
        data = json.loads(res)

        if "streams" not in data or not data["streams"]:
            raise ValueError("No valid video stream found in the input file.")

        stream = data["streams"][0]
        in_w = int(stream["width"])
        in_h = int(stream["height"])

        fps_str = stream["r_frame_rate"]
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            fps = num / den if den != 0 else 0.0
        else:
            fps = float(fps_str)

        return in_w, in_h, fps

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode("utf-8", errors="ignore").strip() if e.stderr else "Unknown error"
        raise ValueError(
            f"ffprobe failed to inspect video '{input_video_path}'. "
            f"Video may be unsupported or corrupted. Details: {stderr_output}"
        ) from e


def upload_to_s3(local_path: str, s3_key: str, job_input: dict) -> str:
    bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")
    endpoint_url = job_input.get("s3_endpoint_url") or os.getenv("S3_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud:443")
    key_id = job_input.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = job_input.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = job_input.get("aws_region") or os.getenv("AWS_REGION", "eu-north1")

    if not bucket:
        raise ValueError("S3_BUCKET was not provided in job payload or environment variables.")

    client = boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        region_name=region,
        endpoint_url=endpoint_url
    )

    client.upload_file(local_path, bucket, s3_key)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600
    )


def process_video_sync(job: dict) -> dict:
    """Synchronous core processing function executed inside the thread executor."""
    job_id = job.get("id", "job")
    job_input = job.get("input", {})

    video_url = job_input.get("video_url")
    if not video_url:
        raise ValueError("Missing 'video_url' key in job input.")

    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"

    try:
        # Step 1: Download (Passes job_input dictionary)
        download_video(job_input, str(input_video))

        # Step 2: Extract details
        in_w, in_h, fps = get_video_info(str(input_video))

        # Step 3: Upscaler logic (placeholder)
        if not output_video.exists():
            output_video = input_video

        # Step 4: Upload result to S3
        output_url = upload_to_s3(str(output_video), f"{job_id}_upscaled.mp4", job_input)

        return {
            "output_url": output_url,
            "width": in_w,
            "height": in_h,
            "fps": fps
        }

    finally:
        # Step 5: Clean up temp directory
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


async def async_handler(job: dict) -> dict:
    """RunPod Serverless entrypoint wrapper."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_video_sync, job)
    return result


# Start the RunPod Serverless listener
runpod.serverless.start({"handler": async_handler})