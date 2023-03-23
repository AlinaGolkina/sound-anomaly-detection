import pathlib
import shutil
import subprocess
import time
from threading import Thread

import fire
import pyaudio
import schedule
from iforest import iforest_preds, iforest_to_onnx
from recorder import record_sound


def train_iforest(target_dir="sound_rec", dir_name_data="train_unlabled"):
    iforest_to_onnx(target_dir, dir_name_data)


def record_single(
    dir_name="sound_rec/train_unlabled",
    channels=1,
    rate=16000,
    frames_per_buffer=1024,
    duration=11,
    input_device_index=None,
):
    record_sound(
        dir_name, channels, rate, frames_per_buffer, duration, input_device_index
    )


def record_two(
    dir_with_wav="sound_rec/record_buffer",
    target_dir="sound_rec/flac",
    channels=1,
    rate=16000,
    frames_per_buffer=1024,
    duration=11,
    input_device_index_1=0,
    input_device_index_2=1,
):
    def thr():
        while True:
            schedule.run_pending()
            time.sleep(1)

    def to_flack(dir_with_wav, target_dir):
        wav_files = list(pathlib.Path(f"{dir_with_wav}").absolute().glob("*.wav"))
        wav_files.sort(key=lambda s: str(s).split("_")[-1])
        for sound in wav_files[:-5]:
            new_name = sound.with_suffix(".flac")
            new_name_short = new_name.name
            if pathlib.Path(target_dir, new_name_short).exists():
                continue
            else:
                comm = f"FFmpeg -i {sound} {new_name}"
                subprocess.call(comm)
                shutil.move(pathlib.Path(dir_with_wav, new_name_short), target_dir)
                pathlib.Path(dir_with_wav, sound).unlink()

    # predictions
    # schedule.every(1).minutes.do(iforest_preds, target_dir="sound_rec",  dir_name_data="/record_buffer", predicted_dir="predicted_records",  onnx_file="train_iforest.onnx",   batch=5)
    # to_flack
    schedule.every(1).minutes.do(to_flack, dir_with_wav, target_dir)
    Thread(target=thr).start()
    # recording first mic (device [0])
    Thread(
        target=record_sound,
        args=(
            dir_with_wav,
            channels,
            rate,
            frames_per_buffer,
            duration,
            input_device_index_1,
        ),
    ).start()
    time.sleep(1)
    # recording second mic (device [1])
    Thread(
        target=record_sound,
        args=(
            dir_with_wav,
            channels,
            rate,
            frames_per_buffer,
            duration,
            input_device_index_2,
        ),
    ).start()


def devices():
    """
    print available devices and device index for pyAudio sound record
    """
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i, p.get_device_info_by_index(i)["name"])


if __name__ == "__main__":
    fire.Fire()
