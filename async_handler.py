import os
import json
import shutil
import subprocess
import requests
import asyncio
import boto3
import torch
import cv2
import numpy as np
import runpod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient, ReturnDocument
from google.oauth2 import service_account
import google.auth.transport.requests

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

executor = ThreadPoolExecutor(max_workers=1)

print("Loading Real-ESRGAN Model into VRAM...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="/app/weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False,
    device=device
)
print("Real-ESRGAN Loaded Successfully!")


def get_video_info(video_path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.check_output(cmd).decode()
    info = json.loads(res)["streams"][0]
    w, h = int(info["width"]), int(info["height"])

    fps_str = info.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(float, fps_str.split("/"))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)

    return w, h, str(fps)


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


def trigger_gcp_reassemble_job(gcp_project: str, gcp_region: str, reassemble_job_name: str, video_id: str, sa_key_json: str):
    """Triggers Cloud Run Job 2 (Reassembler) via Google Cloud REST API."""
    try:
        sa_info = json.loads(sa_key_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        req = google.auth.transport.requests.Request()
        creds.refresh(req)

        url = f"https://run.googleapis.com/v2/projects/{gcp_project}/locations/{gcp_region}/jobs/{reassemble_job_name}:run"
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json"
        }
        body = {
            "overrides": {
                "containerOverrides": [
                    {
                        "env": [
                            {"name": "TARGET_VIDEO_ID", "value": video_id}
                        ]
                    }
                ]
            }
        }
        res = requests.post(url, headers=headers, json=body)
        print(f"Triggered Cloud Run Reassembler Job: {res.status_code} - {res.text}", flush=True)
    except Exception as e:
        print(f"Error triggering GCP Reassembler Job: {e}", flush=True)


def update_atlas_and_check_completion(job_input: dict):
    job_id = job_input["job_id"]
    chunk_index = str(job_input["chunk_index"])
    mongo_uri = job_input["mongo_uri"]

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client.get_database("video_upscaler")
    jobs_col = db.get_collection("jobs")

    # Atomic update: mark chunk COMPLETED & increment completed_count
    updated_doc = jobs_col.find_one_and_update(
        {"_id": job_id},
        {
            "$set": {f"chunks.{chunk_index}.status": "COMPLETED"},
            "$inc": {"completed_count": 1}
        },
        return_document=ReturnDocument.AFTER
    )

    completed_count = updated_doc.get("completed_count", 0)
    total_chunks = updated_doc.get("total_chunks", 0)
    print(f"Job '{job_id}' Progress: {completed_count}/{total_chunks} chunks completed.", flush=True)

    # Trigger reassembly if this worker instance finished the final chunk
    if completed_count >= total_chunks and updated_doc.get("status") != "REASSEMBLING":
        jobs_col.update_one({"_id": job_id}, {"$set": {"status": "REASSEMBLING"}})
        print(f"=== ALL CHUNKS COMPLETED FOR {job_id}! Triggering Reassembler Cloud Run Job... ===", flush=True)
        
        trigger_gcp_reassemble_job(
            gcp_project=job_input["gcp_project"],
            gcp_region=job_input["gcp_region"],
            reassemble_job_name=job_input["reassemble_job_name"],
            video_id=job_id,
            sa_key_json=job_input.get("gcp_sa_key_json", "")
        )


def process_video_sync(job_input: dict) -> dict:
    video_url = job_input.get("video_url")
    requested_scale = job_input.get("scale", 4)
    job_id = job_input.get("job_id", "job")
    chunk_index = job_input.get("chunk_index", 0)

    work_dir = Path(f"/tmp/{job_id}_chunk_{chunk_index}")
    work_dir.mkdir(parents=True, exist_ok=True)

    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"

    try:
        # 1. Download chunk
        res = requests.get(video_url, stream=True)
        with open(input_video, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

        # 2. Extract dimensions and FPS
        in_w, in_h, fps = get_video_info(input_video)

        # 3. Scale calculation capped at 4K
        max_w, max_h = 3840, 2160
        max_allowed_scale = min(max_w / in_w, max_h / in_h)
        effective_scale = min(float(requested_scale), max_allowed_scale)
        effective_scale = max(1.0, effective_scale)

        out_w = int(in_w * effective_scale)
        out_h = int(in_h * effective_scale)

        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

        frame_size = in_w * in_h * 3

        # 4. FFmpeg pipe reader
        reader = subprocess.Popen([
            "ffmpeg", "-v", "error",
            "-i", str(input_video),
            "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # 5. FFmpeg pipe writer (Includes -movflags +faststart)
        writer = subprocess.Popen([
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", fps,
            "-i", "-",
            "-i", str(input_video),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "faster",
            "-movflags", "+faststart",
            "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
            str(output_video)
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # 6. Stream frames through Real-ESRGAN in VRAM
        while True:
            raw_frame = reader.stdout.read(frame_size)
            if not raw_frame or len(raw_frame) != frame_size:
                break

            img = np.frombuffer(raw_frame, dtype=np.uint8).reshape((in_h, in_w, 3))
            output_img, _ = upsampler.enhance(img, outscale=effective_scale)

            if output_img.shape[1] != out_w or output_img.shape[0] != out_h:
                output_img = cv2.resize(output_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.stdin.write(output_img.tobytes())

        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()

        # 7. Upload upscaled chunk directly to target S3 path
        upscaled_s3_key = job_input.get("upscaled_s3_key", f"temp_chunks/{job_id}/upscaled_chunk_{chunk_index:04d}.mp4")
        output_url = upload_to_s3(str(output_video), upscaled_s3_key, job_input)

        # 8. Update MongoDB Atlas & trigger reassembly if last chunk
        update_atlas_and_check_completion(job_input)

        return {"upscaled_video_url": output_url}

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)


async def async_handler(job):
    job_input = job["input"]

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_video_sync, job_input)
    return result


runpod.serverless.start({"handler": async_handler})