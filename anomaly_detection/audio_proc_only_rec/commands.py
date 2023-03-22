import pathlib
import shutil
import subprocess
import time
from datetime import datetime
from threading import Thread
import sys

import fire
import pyaudio
import schedule
from recorder import record_sound



def record_single(
    dir_with_wav="/home/evocargo/audio_proc/sound_rec/record_buffer",
    target_dir="/home/evocargo/audio_proc/sound_rec/flac",
    channels=1,
    rate=44100,
    frames_per_buffer=512,
    duration=60, # 11, 
    input_device_index=None,
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    index = None
    
    # finding device index of USB microphone
    for i in range(0, numdevices):
        if 'USB PnP' in p.get_device_info_by_host_api_device_index(0, i).get('name'):
            index = i
    if index:
        try:
    
            record_sound(dir_with_wav,
                         channels,
                         rate,
                         frames_per_buffer,
                         duration,
                         input_device_index=index)

        except Exception as e:
            print("Error:", e, file=sys.stderr)
            with open("/home/evocargo/audio_proc/recording_errors.txt", "a") as f:
                f.write(f"{timestamp}: Error: {e}\n")
    else:
        with open("/home/evocargo/audio_proc/recording_errors.txt", "a") as f:
            f.write(f"{timestamp}: couldn't find the right microphone\n")
	    
	    
def record_two(
    dir_with_wav="/home/evocargo/audio_proc/sound_rec/record_buffer",
    target_dir="/home/evocargo/audio_proc/sound_rec/flac",
    channels=1,
    rate=44100,
    frames_per_buffer=512,
    duration=11,
    input_device_index_1=None,
    input_device_index_2=None,
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
                comm = f"ffmpeg -i {sound} {new_name}"
                subprocess.call(comm, shell = True)
                shutil.move(pathlib.Path(dir_with_wav, new_name_short), target_dir)
                pathlib.Path(dir_with_wav, sound).unlink()

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
    with open("cron_output2.txt", "w") as f:
	    f.write(f"PyAudio version: {pyaudio.__version__}\n")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i, p.get_device_info_by_index(i)["name"])


if __name__ == "__main__":

	fire.Fire()
