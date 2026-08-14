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

print("Available Environment Variable Keys:", list(os.environ.keys()))
def download_video(url: str, target_path: str) -> None:
    """Downloads an input video with validation to prevent bad or HTML downloads."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to download input video from URL '{url}': {e}") from e

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            f"The provided URL '{url}' returned an HTML page instead of a video. "
            "Please check if the link requires authentication or has expired."
        )

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        raise ValueError(f"Downloaded video file is empty (0 bytes) from URL: {url}")


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

    # Create isolated temp folder for job files
    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"

    try:
        # Step 1: Download
        download_video(video_url, str(input_video))

        # Step 2: Extract details
        in_w, in_h, fps = get_video_info(str(input_video))

        # Step 3: Run your video upscaler logic here
        # Example: upscale_engine(input_path=input_video, output_path=output_video)
        
        # Temporary output placeholder if model writes to output_video
        if not output_video.exists():
            output_video = input_video  # Replace with actual model execution

        # Step 4: Upload result to S3
        output_url = upload_to_s3(str(output_video), f"{job_id}_upscaled.mp4", job_input)

        return {
            "output_url": output_url,
            "width": in_w,
            "height": in_h,
            "fps": fps
        }

    finally:
        # Step 5: Clean up temp directory to save disk space on worker
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


async def async_handler(job: dict) -> dict:
    """RunPod Serverless entrypoint wrapper."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_video_sync, job)
    return result


# Start the RunPod Serverless listener
runpod.serverless.start({"handler": async_handler})