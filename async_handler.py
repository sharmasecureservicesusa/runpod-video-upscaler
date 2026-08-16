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
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from pymongo import MongoClient

executor = ThreadPoolExecutor(max_workers=os.cpu_count())


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
    """Triggers GCP Cloud Run Reassembler Job via REST API."""
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
    """Updates MongoDB chunk state and triggers reassembly if all chunks are done."""
    mongo_uri = job_input.get("mongo_uri") or os.getenv("MONGO_URI")
    job_id = job_input.get("job_id")
    chunk_index = str(job_input.get("chunk_index"))

    if not mongo_uri or not job_id:
        return

    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("video_upscaler")
    jobs_col = db.get_collection("jobs")

    # Atomic update of chunk status and increment completed count
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


def process_video_sync(job: dict) -> dict:
    job_id = job.get("id", "job")
    job_input = job.get("input", {})

    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"

    try:
        # Step 1: Download
        download_video(job_input, str(input_video))

        # Step 2: Info
        in_w, in_h, fps = get_video_info(str(input_video))

        # Step 3: Model processing placeholder (Replace with model call)
        if not output_video.exists():
            output_video = input_video

        # Step 4: Upload result back to S3
        upscaled_s3_key = job_input.get("upscaled_s3_key") or f"temp_chunks/{job_id}/upscaled_chunk.mp4"
        output_url = upload_to_s3(str(output_video), upscaled_s3_key, job_input)

        # Step 5: Update MongoDB & trigger Reassembler if last chunk
        update_mongodb_chunk_status(job_input)

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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, process_video_sync, job)


runpod.serverless.start({"handler": async_handler})