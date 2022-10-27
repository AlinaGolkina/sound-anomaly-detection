import time
from datetime import datetime
from threading import Thread

import schedule
from iforest_test import iforest_preds
from recorder import Recorder


# def first_mic(channels=1, rate=16000, frames_per_buffer=1024, input_device_index=0):
#   rec = Recorder(channels=channels, rate=rate, frames_per_buffer=frames_per_buffer, input_device_index=input_device_index)
#    while True:
#       file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
#      with rec.open(f"{dir_name}/blocking_1mic_{file_name}.wav", "wb") as recfile:
#         recfile.record(duration=11.0)
#        print("recording first mic...")


def second_mic(channels=1, rate=16000, frames_per_buffer=1024, input_device_index=1):
    time.sleep(1)
    rec = Recorder(
        channels=channels,
        rate=rate,
        frames_per_buffer=frames_per_buffer,
        input_device_index=input_device_index,
    )
    while True:
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        with rec.open(f"{dir_name}/blocking_2mic_{file_name}.wav", "wb") as recfile:
            recfile.record(duration=11.0)
        print("recording second mic...")
        time.sleep(1)


def preds():
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
    dir_name = r"sound_rec/record_buffer_2_mics"
    schedule.every(1).minutes.do(preds)
    Thread(target=thr).start()
    # Thread(target=first_mic).start()
    Thread(target=second_mic, args=(1, 16000, 1024, 0)).start()
    time.sleep(1)
    Thread(target=second_mic, args=(1, 16000, 1024, 0)).start()
