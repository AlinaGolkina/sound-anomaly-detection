import time
from threading import Thread

import fire
import pyaudio
import schedule

from .iforest import iforest_preds, iforest_to_onnx
from .recorder import record_sound


def record_single():
    dir_name = "sound_rec/train_unlabled"
    record_sound(dir_name, channels=1, rate=16000, frames_per_buffer=1024)


def record_two():
    def preds():
        """
        get predictions from batch audio samples from dir with records
        per 1 minute( 1 hour)
        """
        iforest_preds(
            target_dir="sound_rec",
            dir_name_data="/record_buffer",
            predicted_dir="predicted_records",
            onnx_file="train_iforest.onnx",
            batch=5,
        )
        print("Прогноз батча аудио раз в минуту")

    def thr():
        while True:
            schedule.run_pending()
            time.sleep(1)

    dir_name = "sound_rec/record_buffer"
    schedule.every(1).minutes.do(iforest_preds, args=(...))
    Thread(target=thr).start()
    # recording first mic (device [0])
    Thread(target=record_sound, args=(dir_name, 1, 16000, 1024, 0)).start()
    time.sleep(1)
    # recording second mic (device [1])
    Thread(target=record_sound, args=(dir_name, 1, 16000, 1024, 1)).start()


def devices():
    """
    print available devices and device index for pyAudio sound record
    """
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i, p.get_device_info_by_index(i)["name"])


if __name__ == "__main__":
    fire.Fire()
