import time
from datetime import datetime
from threading import Thread

import schedule
from iforest_test import iforest_preds
from recorder import Recorder


def record_sound(channels=1, rate=16000, frames_per_buffer=1024, input_device_index=1):
    """
    parameters:
    ----------
        rate – Sampling rate (16000 by default)
        channels – Number of channels (1 by default, mono)
        frames_per_buffer – Specifies the number of frames per buffer
        input_device_index – Index of Input Device to use.
                             Unspecified (or None) uses default device.
                             Ignored if input is False.

    return:
    -------
        10 sec audio samples in dir_name
    """

    rec = Recorder(
        channels=channels,
        rate=rate,
        frames_per_buffer=frames_per_buffer,
        input_device_index=input_device_index,
    )
    while True:
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        with rec.open(
            f"{dir_name}/blocking_{input_device_index}mic_{file_name}.wav", "wb"
        ) as recfile:
            recfile.record(duration=11.0)
        print(f"recording {input_device_index} mic...")


def preds():
    """
    get predictions from batch audio samples from dir with records
    per 1 minute( 1 hour)
    """
    iforest_preds(
        target_dir=r"sound_rec",
        dir_name_data=r"/record_buffer",
        predicted_dir=r"predicted_records",
        onnx_file="train_iforest.onnx",
        batch=5,
    )
    print("Прогноз батча аудио раз в минуту")


def thr():
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    dir_name = r"sound_rec/record_buffer"
    schedule.every(1).minutes.do(preds)
    Thread(target=thr).start()
    # recording first mic (device [0])
    Thread(target=record_sound, args=(1, 16000, 1024, 0)).start()
    time.sleep(1)
    # recording second mic (device [1])
    Thread(target=record_sound, args=(1, 16000, 1024, 1)).start()
