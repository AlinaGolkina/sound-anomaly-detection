# Stream Sound recording and Anomaly detection

### 1. dataset.py

    dataset creation class for anomaly detecting training and test

### 2. device.py

    print available devices and device index for pyAudio sound record

### 3. forest_test.py

    Get predictions for batch of recordered sounds and move files to predicted_dir using pretrained model from  train_iforest.onnx
    
### 4. iforest_train.py

Anomaly detection training with Isolation Forest and saving trained model to
onnx file

### 5. predictions.csv

    csv file with predicted sound batches in format:

    sound_name_file, anomaly/normal
### recorder.py
    PyAudio recorder class 
    Provides WAV recording functionality via two approaches:
    Blocking mode and Non-blocking mode (start and stop recording)
    
    PyAudio documentation: https://people.csail.mit.edu/hubert/pyaudio/docs/
    Recorder class realization by https://gist.github.com/sloria/5693955

### 6. write_sound.py

    sound recording from default input device (if certain device, set parameter
    input_device_index you can get indexes of devices from **device.py**)

### 2. write_sound_2_mics.py

sound recording from two microphones using threading



  


### 5. train_iforest.onnx

onnx file with trained Iforest model


