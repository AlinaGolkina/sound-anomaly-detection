# Stream Sound recording and Anomaly detection
### 1. write_sound.py
sound recording from default input device (if certain device, set parameter input_device_index
you can get indexes of devices from **device.py**)
### 2. write_sound_2_mics.py
sound recording from two microphones using threading
### 3. iforest_train.py
Anomaly detection training with Isolation Forest and saving trained model to onnx file

    parameters:
    ----------
    target_dir: str - sound dir
    dir_name_data: str - dir with train data

    return:
    -------
    write trained iForest model to train_iforest.onnx file
### 4. forest_test.py
    Get predictions for recordered sounds and move files to predicted_dir
### 5. train_iforest.onnx
onnx file with trained Iforest model
### 6. predictions.csv
csv file with predicted sound batches in format:

    sound_name_file, anomaly/normal
### 7. dataset.py
dataset creation class for anomaly detecting

### 8. device.py
print available devices and device index for pyAudio sound record