# Stream Sound recording and Anomaly detection
1. Dirs structure:
   **\sound_rec**
      **\flac** - converted audio records from wavs to flac
      **\predicted_records** 
      **\record_buffer**
      **\train_unlabled**

2. Record the training dataset to **audio_proc/sound_rec/train_unlabled/**

   - **commands.py  record_single** - from one mic and without anomaly prediction thread,

3. Train the iforest model on train dataset (**audio_proc/sound_rec/train_unlabled/**) and saving model to  **train_iforest.onnx**
  - **commands.py  train_iforest**
4. Stream recording audio samples from 2 mics (*commands.py  record_two**) to
   **audio_proc\sound_rec\record_buffer** and converting to .flac format (per 1 minute/hour to
   **\flac**) 

5. requirements in **requirements.txt**

## Files descriptions

| Files                     | Description                                                                                                                                                                                                     |
| :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **dataset.py**            | dataset creation class for anomaly detecting (for training and test dataset creation)                                                                                                                                                                                                 |
| **iforest.py**            | training iforest model and predictions   |     
| **predictions.csv**       | csv file with predicted sound batches in format: sound_name_file, anomaly/normal                                              |
| **recorder.py**           | PyAudio recorder class, provides WAV recording functionality, PyAudio documentation: [https://people.csail.mit.edu/hubert/pyaudio/docs/] | Recorder class realization by [https://gist.github.com/sloria/5693955] |
| **train_iforest.onnx**    | trained iForest model          |                                                                                                         
| **commands.py**           | commands for sound recording (1 mic and 2 mics), iforest model training and getting device index)  |                                     
| **requirements.txt**      | dependencies                                                                                                                           |
