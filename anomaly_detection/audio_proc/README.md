# Stream Sound recording and Anomaly detection


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
