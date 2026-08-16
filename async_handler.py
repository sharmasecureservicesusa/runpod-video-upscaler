import asyncio
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import cv2
import requests
import runpod
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from pymongo import MongoClient
from realesrgan import RealESRGANer

# Thread pool for CPU/GPU heavy operations
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# ==============================================================================
# Global AI Model Initialization (Loaded once during Container Cold Start)
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing Real-ESRGAN AI Model on device: {device}", flush=True)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upscaler = RealESRGANer(
    scale=4,
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    model=model,
    tile=400,         # Prevents CUDA Out-Of-Memory (OOM) errors on large frames
    tile_pad=10,
    pre_pad=0,
    half=True if device.type == "cuda" else False,
    device=device
)
print("Real-ESRGAN AI Model loaded into GPU memory successfully.", flush=True)


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_s3_client(job_input: dict):
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
    video_url = job_input.get("video_url", "")
    if not video_url:
        raise ValueError("Missing 'video_url' in job_input.")

    s3_bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")

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
        print(f"Downloading via HTTP GET: {video_url[:80]}...", flush=True)
        res = requests.get(video_url, stream=True)
        res.raise_for_status()
        with open(local_destination, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

    if not os.path.exists(local_destination) or os.path.getsize(local_destination) == 0:
        raise ValueError(f"Downloaded video file is empty from URL: {video_url}")


def get_video_info(input_video_path: str):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(input_video_path)
    ]
    res = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8")
    data = json.loads(res)
    stream = data["streams"][0]
    in_w = int(stream["width"])
    in_h = int(stream["height"])

    fps_str = stream["r_frame_rate"]
    num, den = map(float, fps_str.split("/")) if "/" in fps_str else (float(fps_str), 1.0)
    fps = num / den if den != 0 else 0.0

    return in_w, in_h, fps


def calculate_4k_dimensions(in_w: int, in_h: int, target_scale: float = 4.0):
    """Calculates dimensions capped at 3840x2160, keeping even pixel values."""
    MAX_W, MAX_H = 3840, 2160

    scale_w = MAX_W / in_w
    scale_h = MAX_H / in_h
    final_scale = min(target_scale, scale_w, scale_h)

    if final_scale < 1.0:
        final_scale = 1.0

    target_w = int(in_w * final_scale) // 2 * 2
    target_h = int(in_h * final_scale) // 2 * 2

    return target_w, target_h, final_scale


def upscale_video_realesrgan_4k(input_path: str, output_path: str, target_scale: float = 4.0):
    """Option B: AI Real-ESRGAN Frame-by-Frame GPU Upscaling (4K Max Cap)."""
    in_w, in_h, fps = get_video_info(input_path)
    target_w, target_h, final_scale = calculate_4k_dimensions(in_w, in_h, target_scale)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_raw_output = str(Path(output_path).parent / "temp_ai_raw.mp4")
    out = cv2.VideoWriter(temp_raw_output, fourcc, fps, (target_w, target_h))

    print(f"Starting Real-ESRGAN AI upscale: {in_w}x{in_h} -> {target_w}x{target_h} @ {fps}fps", flush=True)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # AI Frame enhancement
        output_frame, _ = upscaler.enhance(frame, outscale=final_scale)

        # Enforce exact dimension bounds if rounding differs by a pixel
        if output_frame.shape[1] != target_w or output_frame.shape[0] != target_h:
            output_frame = cv2.resize(output_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        out.write(output_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"AI Upscaled {frame_idx} frames...", flush=True)

    cap.release()
    out.release()

    # Re-encode with FFmpeg to H.264 and copy original audio track
    print("Re-encoding AI frames with FFmpeg for browser compatibility...", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", temp_raw_output,
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "faster",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_path)
    ], check=True)

    if os.path.exists(temp_raw_output):
        os.remove(temp_raw_output)

    return target_w, target_h


def upload_to_s3(local_path: str, s3_key: str, job_input: dict) -> str:
    bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")
    s3_client = get_s3_client(job_input)
    s3_client.upload_file(local_path, bucket, s3_key)

    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600
    )


def trigger_gcp_reassemble_job(gcp_project: str, gcp_region: str, job_name: str, job_id: str):
    url = f"https://{gcp_region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{gcp_project}/jobs/{job_name}:run"
    auth_req = Request()
    token = id_token.fetch_id_token(auth_req, "https://run.googleapis.com/")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "overrides": {
            "containerOverrides": [{
                "env": [{"name": "JOB_ID", "value": job_id}]
            }]
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    print(f"Triggered GCP Reassembler Job '{job_name}' for Job ID '{job_id}': Status {res.status_code}", flush=True)


def update_mongodb_chunk_status(job_input: dict):
    mongo_uri = job_input.get("mongo_uri") or os.getenv("MONGO_URI")
    job_id = job_input.get("job_id")
    chunk_index = str(job_input.get("chunk_index"))

    if not mongo_uri or not job_id:
        return

    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("video_upscaler")
    jobs_col = db.get_collection("jobs")

    updated_doc = jobs_col.find_one_and_update(
        {"_id": job_id},
        {
            "$set": {f"chunks.{chunk_index}.status": "COMPLETED"},
            "$inc": {"completed_count": 1}
        },
        return_document=True
    )

    if updated_doc:
        completed = updated_doc.get("completed_count", 0)
        total = updated_doc.get("total_chunks", 0)
        print(f"Job '{job_id}': Progress {completed}/{total} chunks completed.", flush=True)

        if completed >= total and updated_doc.get("status") != "READY_FOR_REASSEMBLE":
            jobs_col.update_one({"_id": job_id}, {"$set": {"status": "READY_FOR_REASSEMBLE"}})

            gcp_project = job_input.get("gcp_project") or os.getenv("GCP_PROJECT_ID")
            gcp_region = job_input.get("gcp_region") or os.getenv("GCP_REGION", "us-central1")
            reassemble_job = job_input.get("reassemble_job_name") or os.getenv("REASSEMBLE_JOB_NAME", "upscale-reassemble-job")

            if gcp_project and reassemble_job:
                try:
                    trigger_gcp_reassemble_job(gcp_project, gcp_region, reassemble_job, job_id)
                except Exception as e:
                    print(f"Failed to trigger Reassembler job: {e}", flush=True)


# ==============================================================================
# Synchronous Core Function Execution
# ==============================================================================
def process_video_sync(job: dict) -> dict:
    job_id = job.get("id", "job")
    job_input = job.get("input", {})
    scale = float(job_input.get("scale", 4.0))

    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"

    try:
        # Step 1: Download raw chunk from Amazon S3
        download_video(job_input, str(input_video))

        # Step 2: Extract details
        _, _, fps = get_video_info(str(input_video))

        # Step 3: Run Option B (Real-ESRGAN AI GPU Upscaling with 4K Cap)
        out_w, out_h = upscale_video_realesrgan_4k(str(input_video), str(output_video), target_scale=scale)

        # Step 4: Upload upscaled chunk back to Amazon S3
        upscaled_s3_key = job_input.get("upscaled_s3_key") or f"temp_chunks/{job_id}/upscaled_chunk.mp4"
        output_url = upload_to_s3(str(output_video), upscaled_s3_key, job_input)

        # Step 5: Update MongoDB & auto-trigger Cloud Run Job 2 if last chunk
        update_mongodb_chunk_status(job_input)

        return {
            "output_url": output_url,
            "width": out_w,
            "height": out_h,
            "fps": fps
        }

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


async def async_handler(job: dict) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, process_video_sync, job)


runpod.serverless.start({"handler": async_handler})