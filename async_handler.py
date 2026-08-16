import asyncio
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
import boto3
import requests
import runpod

# Thread executor to handle blocking CPU/GPU tasks without freezing asyncio
executor = ThreadPoolExecutor(max_workers=os.cpu_count())


def get_s3_client(job_input: dict):
    """Creates a standard AWS S3 client using credentials from input or environment."""
    key_id = job_input.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = job_input.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = job_input.get("aws_region") or os.getenv("AWS_REGION", "us-east-1")

    return boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        region_name=region
    )


def download_video(job_input: dict, local_destination: str):
    """
    Downloads video chunk from Amazon S3 or a presigned/public HTTP URL.
    """
    video_url = job_input.get("video_url", "")
    if not video_url:
        raise ValueError("Missing 'video_url' in job_input.")

    s3_bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")

    # If it's an S3 URI or contains s3.amazonaws.com / bucket name without presigned query
    if video_url.startswith("s3://") or (s3_bucket and s3_bucket in video_url and "?" not in video_url):
        if video_url.startswith("s3://"):
            parts = video_url.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            s3_key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = s3_bucket
            s3_key = video_url.split(f"{bucket}/")[-1] if f"{bucket}/" in video_url else urlparse(video_url).path.lstrip("/")

        print(f"Downloading directly from Amazon S3: bucket='{bucket}', key='{s3_key}'", flush=True)
        s3_client = get_s3_client(job_input)
        s3_client.download_file(bucket, s3_key, local_destination)

    else:
        # Download presigned or public HTTP URL
        print(f"Downloading via HTTP GET: {video_url[:80]}...", flush=True)
        res = requests.get(video_url, stream=True)
        res.raise_for_status()
        with open(local_destination, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

    if not os.path.exists(local_destination) or os.path.getsize(local_destination) == 0:
        raise ValueError(f"Downloaded video file is empty (0 bytes) from URL: {video_url}")


def get_video_info(input_video_path: str):
    """Probes video details safely using ffprobe."""
    if not os.path.exists(input_video_path) or os.path.getsize(input_video_path) == 0:
        raise ValueError(f"Input video file missing or empty at: {input_video_path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(input_video_path)
    ]

    try:
        res = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8")
        data = json.loads(res)

        if "streams" not in data or not data["streams"]:
            raise ValueError("No valid video stream found.")

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
        raise ValueError(f"ffprobe failed: {stderr_output}") from e


def upload_to_s3(local_path: str, s3_key: str, job_input: dict) -> str:
    """Uploads processing output back to Amazon S3 and generates a presigned GET URL."""
    bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET was not provided in job payload or environment variables.")

    s3_client = get_s3_client(job_input)
    s3_client.upload_file(local_path, bucket, s3_key)

    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600
    )


def process_video_sync(job: dict) -> dict:
    """Synchronous core processing function."""
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
        # Step 1: Download
        download_video(job_input, str(input_video))

        # Step 2: Extract info
        in_w, in_h, fps = get_video_info(str(input_video))

        # Step 3: Model processing (placeholder output)
        if not output_video.exists():
            output_video = input_video

        # Step 4: Upload result back to S3
        upscaled_s3_key = job_input.get("upscaled_s3_key") or f"temp_chunks/{job_id}/upscaled_chunk.mp4"
        output_url = upload_to_s3(str(output_video), upscaled_s3_key, job_input)

        return {
            "output_url": output_url,
            "width": in_w,
            "height": in_h,
            "fps": fps
        }

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


async def async_handler(job: dict) -> dict:
    """RunPod Serverless entrypoint wrapper."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, process_video_sync, job)


runpod.serverless.start({"handler": async_handler})