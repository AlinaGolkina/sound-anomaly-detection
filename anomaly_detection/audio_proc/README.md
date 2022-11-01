# Stream Sound recording and Anomaly detection

1. Record the training dataset to **audio_proc/sound_rec/train_unlabled/**  
     - **write_sound.py** - from one mic and without anomaly prediction thread, 
     
2. Train the iforest model (**iforest_train.py**) on train dataset (**audio_proc/sound_rec/train_unlabled/**) and saving model to **train_iforest.onnx** 

3. Stream recording audio samples from 2 mics (**write_sound_2_mics.py**) to **audio_proc\sound_rec\record_buffer** and anomaly prediction of batches of audio samples from recorded sounds buffer (per 1 minute/hour to **predictions.csv**) and moving predicted audio samples to **audio_proc/sound_rec/predicted_records/**

4. requirements in **requirements.txt**:
- librosa==0.9.2
- numpy==1.23.2
- onnxruntime==1.12.1
- PyAudio==0.2.12
- schedule==1.1.0
- scikit_learn==1.1.3
- scipy==1.8.1
- skl2onnx==1.13
- torch==1.12.1
- tqdm==4.64.0

## Files descriptions


Files                      | Description                                                   
:--------                 |:-----                                                          
**dataset.py**            | dataset creation class for anomaly detecting (for training and test dataset creation)
**device.py**             | print available devices and device index for pyAudio sound record
**forest_test.py**        | Get predictions for batch of recordered sounds and move files to predicted_dir using pretrained model from  train_iforest.onnx
**iforest_train.py**      | Anomaly detection training with Isolation Forest and saving trained model to onnx file
**predictions.csv**       | csv file with predicted sound batches in format: sound_name_file, anomaly/normal
**recorder.py**           |PyAudio recorder class, provides WAV recording functionality, PyAudio documentation: [https://people.csail.mit.edu/hubert/pyaudio/docs/] Recorder class realization by [https://gist.github.com/sloria/5693955]
**train_iforest.onnx**    |trained iForest model
**write_sound.py**        |sound recording from default input device (if certain device, set parameter input_device_index, you can get indexes of devices from **device.py**)
**write_sound_2_mics.py** |sound recording from two microphones using threading, and anomaly predictions for batch recorded audio samples per 1 hour
**requirements.txt**      |dependencies
