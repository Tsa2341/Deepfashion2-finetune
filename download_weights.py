import os
import sys
import time
from typing import Optional

import requests


def download_weights(url: str, dest_dir: str, chunk_size: int = 262144) -> str:
    """Download a file from `url` into `dest_dir` and return the saved path.

    Shows simple progress (MiB and instantaneous MiB/s). Skips download if file
    already exists with the same size when available.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url.split("?")[0])
    dest_path = os.path.join(dest_dir, filename)

    with requests.Session() as s:
        resp = s.get(url, stream=True)
        resp.raise_for_status()

        total = resp.headers.get("Content-Length")
        try:
            total = int(total) if total is not None else None
        except Exception:
            total = None

        # If file exists and size matches, skip
        if os.path.exists(dest_path) and total is not None:
            if os.path.getsize(dest_path) == total:
                print(f"Weights already present at {dest_path}, skipping download")
                return dest_path

        downloaded = 0
        last_print = 0
        start_time = time.time()
        last_time = start_time
        last_bytes = 0

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded * 100.0 / total
                    now = time.time()
                    interval = max(now - last_time, 1e-6)
                    bytes_interval = downloaded - last_bytes
                    inst_speed_mib_s = (bytes_interval / interval) / (1024 * 1024)
                    if percent - last_print >= 0.5 or downloaded == total:
                        dl_mb = downloaded / (1024 * 1024)
                        tot_mb = total / (1024 * 1024)
                        print(
                            f"\rDownloading weights: {percent:5.2f}% ({dl_mb:6.2f}/{tot_mb:6.2f} MiB) @ {inst_speed_mib_s:5.2f} MiB/s",
                            end="",
                            flush=True,
                        )
                        last_print = percent
                        last_time = now
                        last_bytes = downloaded
                else:
                    if downloaded - last_print >= 1024 * 1024:
                        now = time.time()
                        interval = max(now - last_time, 1e-6)
                        bytes_interval = downloaded - last_bytes
                        inst_speed_mib_s = (bytes_interval / interval) / (1024 * 1024)
                        dl_mb = downloaded / (1024 * 1024)
                        print(
                            f"\rDownloading weights: {dl_mb:6.2f} MiB @ {inst_speed_mib_s:5.2f} MiB/s",
                            end="",
                            flush=True,
                        )
                        last_print = downloaded
                        last_time = now
                        last_bytes = downloaded

    # final summary
    elapsed = max(time.time() - start_time, 1e-6)
    avg_mib_s = (downloaded / elapsed) / (1024 * 1024)
    dl_mb = downloaded / (1024 * 1024)
    if total:
        tot_mb = total / (1024 * 1024)
        print(
            f"\rDownloaded weights to {dest_path}: {dl_mb:6.2f}/{tot_mb:6.2f} MiB @ {avg_mib_s:5.2f} MiB/s"
        )
    else:
        print(
            f"\rDownloaded weights to {dest_path}: {dl_mb:.2f} MiB @ {avg_mib_s:5.2f} MiB/s"
        )

    return dest_path


def main(argv: Optional[list] = None):
    argv = argv or sys.argv[1:]
    if len(argv) >= 1:
        url = argv[0]
    else:
        url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth"
    dest = os.path.join(os.path.dirname(__file__), "weights")
    download_weights(url, dest)


if __name__ == "__main__":
    main()
