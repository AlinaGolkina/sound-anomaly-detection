import pyaudio

def devices():
    '''
    print available devices and device index for pyAudio sound record
    '''
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i, p.get_device_info_by_index(i)['name'])

if __name__ == "__main__":
    devices()