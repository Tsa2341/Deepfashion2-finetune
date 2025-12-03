import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import re
import time
import requests
import zipfile
from typing import Optional

from download_weights import download_weights
from deepfashion2_to_coco import convert_deepfashion2_to_coco


def download_file_from_google_drive(url: str, dest_path: str) -> None:
    """Download a file from a Google Drive share URL to `dest_path`.

    Handles both the small-file direct download and the large-file "can't scan"
    confirmation form. Streams to disk and prints progress with instantaneous
    speed (MiB/s) computed over the most recent print interval.
    """
    session = requests.Session()

    # Extract file id
    file_id = None
    m = re.search(r"/d/([0-9A-Za-z_-]+)", url)
    if m:
        file_id = m.group(1)
    else:
        m = re.search(r"[?&]id=([0-9A-Za-z_-]+)", url)
        if m:
            file_id = m.group(1)

    if not file_id:
        raise ValueError(f"Could not extract Google Drive file id from URL: {url}")

    def is_zip_file(path: str) -> bool:
        try:
            with open(path, "rb") as f:
                sig = f.read(4)
                return sig.startswith(b"PK\x03\x04")
        except Exception:
            return False

    if os.path.exists(dest_path) and is_zip_file(dest_path):
        print(f"Destination exists and is a zip, skipping download: {dest_path}")
        return

    if os.path.exists(dest_path) and not is_zip_file(dest_path):
        print(f"Destination exists but is not a zip; will re-download: {dest_path}")
        try:
            os.remove(dest_path)
        except Exception:
            pass

    base_url = "https://drive.google.com/uc?export=download"

    def get_confirm_token(resp_text, resp_cookies):
        # Cookie-based token
        for k, v in resp_cookies.items():
            if k.startswith("download_warning"):
                return v

        m = re.search(r"confirm=([0-9A-Za-z_-]+)&", resp_text)
        if m:
            return m.group(1)

        m = re.search(
            r"<input[^>]+name=[\"']confirm[\"'][^>]+value=[\"']([^\"']+)[\"']",
            resp_text,
        )
        if m:
            return m.group(1)

        m = re.search(r'"confirm"\s*:\s*"([0-9A-Za-z_-]+)"', resp_text)
        if m:
            return m.group(1)

        return None

    def save_response_content(
        response: requests.Response, destination, chunk_size=262144
    ):
        total = response.headers.get("Content-Length")
        try:
            total = int(total) if total is not None else None
        except Exception:
            total = None

        with open(destination, "wb") as f:
            downloaded = 0
            last_print = 0
            start_time = time.time()
            last_time = start_time
            last_bytes = 0

            for chunk in response.iter_content(chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                # When total is known print percent every ~0.5% (or at completion)
                if total:
                    percent = downloaded * 100.0 / total
                    now = time.time()
                    interval = max(now - last_time, 1e-6)
                    bytes_interval = downloaded - last_bytes
                    inst_speed_bps = bytes_interval / interval
                    inst_speed_mib_s = inst_speed_bps / (1024 * 1024)

                    if percent - last_print >= 0.5 or downloaded == total:
                        dl_mb = downloaded / (1024 * 1024)
                        tot_mb = total / (1024 * 1024)
                        print(
                            f"\rDownloading {os.path.basename(destination)}: {percent:5.2f}% ({dl_mb:6.2f} MiB / {tot_mb:6.2f} MiB) @ {inst_speed_mib_s:5.2f} MiB/s",
                            end="",
                            flush=True,
                        )
                        last_print = percent
                        last_time = now
                        last_bytes = downloaded
                else:
                    # Unknown total: print every 1 MiB
                    if downloaded - last_print >= 1024 * 1024:
                        now = time.time()
                        interval = max(now - last_time, 1e-6)
                        bytes_interval = downloaded - last_bytes
                        inst_speed_bps = bytes_interval / interval
                        inst_speed_mib_s = inst_speed_bps / (1024 * 1024)
                        dl_mb = downloaded / (1024 * 1024)
                        print(
                            f"\rDownloading {os.path.basename(destination)}: {dl_mb:.2f} MiB @ {inst_speed_mib_s:5.2f} MiB/s",
                            end="",
                            flush=True,
                        )
                        last_print = downloaded
                        last_time = now
                        last_bytes = downloaded

        # Final summary (average speed)
        now = time.time()
        elapsed = max(now - start_time, 1e-6)
        avg_speed_bps = downloaded / elapsed
        avg_speed_mib_s = avg_speed_bps / (1024 * 1024)

        if total:
            dl_mb = downloaded / (1024 * 1024)
            tot_mb = total / (1024 * 1024)
            print(
                f"\rDownloaded {os.path.basename(destination)}: 100.00% ({dl_mb:6.2f} MiB / {tot_mb:6.2f} MiB) @ {avg_speed_mib_s:5.2f} MiB/s"
            )
        else:
            dl_mb = downloaded / (1024 * 1024)
            print(
                f"\rDownloaded {os.path.basename(destination)}: {dl_mb:.2f} MiB @ {avg_speed_mib_s:5.2f} MiB/s"
            )

    # Initial request
    params = {"id": file_id}
    resp = session.get(base_url, params=params, stream=True)

    # If Drive returns an HTML confirmation page ("can't scan this file"), follow the form/action
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        page = resp.text
        m_action = re.search(r"<form[^>]+action=[\"']([^\"']+)[\"']", page)
        inputs = {}
        for m in re.finditer(r"<input[^>]+type=[\"']hidden[\"'][^>]*>", page):
            inp = m.group(0)
            name_m = re.search(r"name=[\"']([^\"']+)[\"']", inp)
            val_m = re.search(r"value=[\"']([^\"']+)[\"']", inp)
            if name_m:
                inputs[name_m.group(1)] = val_m.group(1) if val_m else ""

        if m_action:
            action_url = m_action.group(1)
            if action_url.startswith("/"):
                action_url = "https://drive.google.com" + action_url

            if inputs:
                resp.close()
                resp = session.get(action_url, params=inputs, stream=True)
            else:
                token = get_confirm_token(page, resp.cookies)
                if token:
                    params["confirm"] = token
                    resp.close()
                    resp = session.get(base_url, params=params, stream=True)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download file: HTTP {resp.status_code}")

    final_ct = resp.headers.get("Content-Type", "")
    if "text/html" in final_ct:
        debug_html = dest_path + ".html"
        with open(debug_html, "wb") as f:
            f.write(resp.content)
        raise RuntimeError(
            f"Download returned HTML (Google Drive confirmation page). Saved HTML to {debug_html}."
            " Try opening the file in a browser to accept the download, or install aria2/gdown."
        )

    save_response_content(resp, dest_path)


def unzip_file(zip_path: str, extract_to: str, password: Optional[str] = None) -> None:
    # Extract with progress reporting. We compute total uncompressed size
    # and stream each member to disk while updating a progress line.
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        if password:
            pwd = password.encode()
        else:
            pwd = None

        members = zip_ref.infolist()
        total = sum(m.file_size for m in members)

        if total == 0:
            # Fallback to extractall if sizes are not available
            if pwd:
                zip_ref.setpassword(pwd)
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} (no size information)")
            return

        extracted = 0
        last_print = 0
        start_time = time.time()
        last_time = start_time
        last_bytes = 0

        for m in members:
            # Build target path and ensure directories exist
            target_path = os.path.join(extract_to, m.filename)
            if m.is_dir():
                os.makedirs(target_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with zip_ref.open(m, pwd=pwd) as src, open(target_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 64)
                    if not chunk:
                        break
                    dst.write(chunk)
                    extracted += len(chunk)

                    # progress update
                    percent = extracted * 100.0 / total
                    now = time.time()
                    interval = max(now - last_time, 1e-6)
                    bytes_interval = extracted - last_bytes
                    inst_speed_bps = bytes_interval / interval
                    inst_speed_mib_s = inst_speed_bps / (1024 * 1024)

                    if percent - last_print >= 0.5 or extracted == total:
                        dl_mb = extracted / (1024 * 1024)
                        tot_mb = total / (1024 * 1024)
                        print(
                            f"\rUnzipping {os.path.basename(zip_path)}: {percent:5.2f}% ({dl_mb:6.2f} MiB / {tot_mb:6.2f} MiB) @ {inst_speed_mib_s:5.2f} MiB/s",
                            end="",
                            flush=True,
                        )
                        last_print = percent
                        last_time = now
                        last_bytes = extracted

        # final summary
        now = time.time()
        elapsed = max(now - start_time, 1e-6)
        avg_speed_bps = extracted / elapsed
        avg_speed_mib_s = avg_speed_bps / (1024 * 1024)
        dl_mb = extracted / (1024 * 1024)
        tot_mb = total / (1024 * 1024)
        print(
            f"\rUnzipped {os.path.basename(zip_path)}: 100.00% ({dl_mb:6.2f} MiB / {tot_mb:6.2f} MiB) @ {avg_speed_mib_s:5.2f} MiB/s"
        )


def download_and_unzip_from_drive_folder() -> None:
    train_zip_url = "https://drive.google.com/file/d/1i9ijLrrYi5lnR0iPhd-lOsxwYzslKdGZ/view?usp=sharing"
    validation_zip_url = "https://drive.google.com/file/d/1i9ijLrrYi5lnR0iPhd-lOsxwYzslKdGZ/view?usp=sharing"

    DEST_DIR = "./datasets/deepfashion2"
    PASSWORD = "2019Deepfashion2**"

    os.makedirs(DEST_DIR, exist_ok=True)

    train_zip_path = os.path.join(DEST_DIR, "deepfashion2_train.zip")
    validation_zip_path = os.path.join(DEST_DIR, "deepfashion2_validation.zip")

    download_file_from_google_drive(train_zip_url, train_zip_path)
    unzip_file(train_zip_path, DEST_DIR, PASSWORD)

    download_file_from_google_drive(validation_zip_url, validation_zip_path)
    unzip_file(validation_zip_path, DEST_DIR, PASSWORD)


def main():
    # Ensure pretrained weights are available
    weights_url = (
        "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
        "swin_base_patch4_window12_384.pth"
    )
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    download_weights(weights_url, weights_dir)

    # Download DeepFashion2 dataset from Google Drive folder and unzip it
    download_and_unzip_from_drive_folder()

    # Convert annotations to COCO format
    DEST_DIR = "./datasets/deepfashion2"
    train_annos_dir = os.path.join(DEST_DIR, "train/annos")
    train_images_dir = os.path.join(DEST_DIR, "train/image")
    train_json = os.path.join(DEST_DIR, "deepfashion2_train.json")
    val_annos_dir = os.path.join(DEST_DIR, "validation/annos")
    val_images_dir = os.path.join(DEST_DIR, "validation/image")
    val_json = os.path.join(DEST_DIR, "deepfashion2_val.json")
    convert_deepfashion2_to_coco(train_annos_dir, train_images_dir, train_json)
    convert_deepfashion2_to_coco(val_annos_dir, val_images_dir, val_json)


if __name__ == "__main__":
    main()
