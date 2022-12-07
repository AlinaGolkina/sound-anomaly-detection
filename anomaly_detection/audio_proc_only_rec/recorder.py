"""
    Code borrowed from:
    https://people.csail.mit.edu/hubert/pyaudio/docs/
    https://gist.github.com/sloria/5693955
    Provides WAV recording functionality via two approaches:
    Blocking mode (record for a set duration):
        rec = Recorder(channels=2)
        with rec.open('blocking.wav', 'wb') as recfile:
            recfile.record(duration=5.0)
    Non-blocking mode (start and stop recording):
        rec = Recorder(channels=2)
        with rec.open('nonblocking.wav', 'wb') as recfile2:
            recfile2.start_recording()
            time.sleep(5.0)
            recfile2.stop_recording()
"""
import wave
from datetime import datetime

import pyaudio


class Recorder(object):
    """A recorder class for recording audio to a WAV file.
    Records in mono by default.
    """

    def __init__(self, channels, rate, frames_per_buffer, input_device_index):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.input_device_index = input_device_index

    def open(self, fname, mode="wb"):
        """Some useful info"""
        return RecordingFile(
            fname,
            mode,
            self.channels,
            self.rate,
            self.frames_per_buffer,
            self.input_device_index,
        )


class RecordingFile:
    def __init__(
        self, fname, mode, channels, rate, frames_per_buffer, input_device_index
    ):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.input_device_index = input_device_index

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration: float):
        """Use a stream with no callback function in blocking mode

        Args:
            duration: seconds
        """
        self._stream = self._pa.open(
            self.rate,
            self.channels,
            pyaudio.paInt16,
            True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.frames_per_buffer,
        )
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)

    def start_recording(self):
        """Use a stream with a callback in non-blocking mode"""
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.get_callback(),
        )
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue

        return callback

    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode="wb"):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile


def record_sound(
    dir_name, channels, rate, frames_per_buffer, duration, input_device_index
):
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
    rec = Recorder(channels, rate, frames_per_buffer, input_device_index)
    while True:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"{dir_name}/blocking_{input_device_index}mic_{timestamp}.wav"
        with rec.open(fname, "wb") as recfile:
            recfile.record(duration)
        print(f"recording {input_device_index} mic...")
