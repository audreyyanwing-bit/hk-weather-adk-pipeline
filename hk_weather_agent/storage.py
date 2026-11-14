"""Step 4 - Storage: upload raw data, processed data and the dashboard to Google Cloud Storage.

Authentication uses Application Default Credentials (run
`gcloud auth application-default login` once), so no key file is needed
in the repository.
"""

from pathlib import Path

from google.adk.tools import ToolContext

from . import config

CONTENT_TYPES = {".json": "application/json", ".csv": "text/csv", ".html": "text/html"}


def upload_to_cloud_storage(tool_context: ToolContext) -> dict:
    """Uploads every file from this pipeline run to the Google Cloud Storage bucket.

    Files are stored under gs://<bucket>/<prefix>/<run_id>/raw/, /processed/
    and dashboard.html, so each run is kept separately.

    Returns:
        Status, the bucket location and the list of uploaded object URIs.
    """
    run_dir = tool_context.state.get("run_dir")
    run_id = tool_context.state.get("run_id")
    if not run_dir or not run_id:
        return {"status": "error", "message": "Nothing to upload - earlier pipeline steps did not run."}

    files = sorted(p for p in Path(run_dir).rglob("*") if p.is_file())
    if not config.GCS_BUCKET:
        return {
            "status": "skipped",
            "message": "GCS_BUCKET is not set in .env, so files were kept locally only.",
            "local_folder": run_dir,
            "file_count": len(files),
        }

    try:
        from google.cloud import storage

        client = storage.Client(project=config.GCP_PROJECT)
        bucket = client.bucket(config.GCS_BUCKET)
        uploaded = []
        for path in files:
            rel = path.relative_to(run_dir).as_posix()
            blob_name = f"{config.GCS_PREFIX}/{run_id}/{rel}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(path), content_type=CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
            uploaded.append(f"gs://{config.GCS_BUCKET}/{blob_name}")
    except Exception as err:  # surface auth/permission/network problems to the agent
        return {
            "status": "error",
            "message": f"Upload to gs://{config.GCS_BUCKET} failed: {type(err).__name__}: {err}",
            "hint": "Check GCS_BUCKET and GOOGLE_CLOUD_PROJECT in .env and run `gcloud auth application-default login`.",
            "local_folder": run_dir,
        }

    tool_context.state["gcs_uris"] = uploaded
    return {
        "status": "success",
        "bucket_location": f"gs://{config.GCS_BUCKET}/{config.GCS_PREFIX}/{run_id}/",
        "uploaded_count": len(uploaded),
        "uploaded": uploaded,
    }
