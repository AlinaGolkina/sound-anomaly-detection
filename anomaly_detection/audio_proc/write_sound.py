from datetime import datetime

from recorder import Recorder


def record_sound(channels=1, rate=16000, frames_per_buffer=1024, input_device_index=0):
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
        with rec.open(f"{dir_name}/blocking_{file_name}.wav", "wb") as recfile:
            recfile.record(duration=11.0)
        print(f"recording...")


if __name__ == "__main__":
    dir_name = r"sound_rec/record_buffer"
    # record from 1 mic
    record_sound(channels=1, rate=16000, frames_per_buffer=1024, input_device_index=0)
