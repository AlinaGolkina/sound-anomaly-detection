import subprocess
from datetime import datetime
from pathlib import Path

import requests


def move_flack_to_s3(flac_dir="/home/evocargo/audio_proc/sound_rec/flac"):
    folder_name = datetime.now().strftime("%Y%m%d")
    if any(Path(flac_dir).iterdir()):
        subprocess.run(
            [
                "aws",
                f"--endpoint-url=https://storage.yandexcloud.net",
                "s3",
                "mv",
                f"--recursive",
                flac_dir,
                f"s3://evo-audio-data/{folder_name}/",
            ]
        )


def move_logs_to_s3():
    if Path("/home/evocargo/audio_proc/recording_errors.txt").exists():
        subprocess.run(
            [
                "aws",
                f"--endpoint-url=https://storage.yandexcloud.net",
                "s3",
                "cp",
                "/home/evocargo/audio_proc/recording_errors.txt",
                f"s3://evo-audio-data/logs/",
            ]
        )


def is_cnx_active(timeout=5):
    try:
        requests.head("https://storage.yandexcloud.net/", timeout=timeout)
        return True
    except requests.ConnectionError:
        return False


while True:
    if is_cnx_active() is True:
        # The internet connection is active
        move_flack_to_s3()
        move_logs_to_s3()
        break
    else:
        pass
