# Stream Sound recording and Anomaly detection

### Dirs structure:

\sound_rec

        \flac - converted audio records from wavs to flac

        \record_buffer - recorded audio samples from 2 mics

        \train_unlabled - recorded audio samples from 1 mic

### Stream recording audio samples from 2 mics

- **commands.py record_two** - from two mics record to
  **sound_rec\record_buffer** and converting to .flac format (per 1 hour to
  **sound_rec\flac**)

### Record the training dataset from 1 mic

- **commands.py record_single** - from one mic to **sound_rec/train_unlabled/**

### requirements:

- **requirements.txt**

### Files descriptions

| Files                | Description                                                                                                                              |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **recorder.py**      | PyAudio recorder class, provides WAV recording functionality, PyAudio documentation: [https://people.csail.mit.edu/hubert/pyaudio/docs/] | Recorder class realization by [https://gist.github.com/sloria/5693955] |
| **commands.py**      | commands for sound recording (1 mic and 2 mics), convert to flack and getting device index)                                              |
| **requirements.txt** | dependencies                                                                                                                             |

# Record process for docker

docker build . -t audioproc

### for record_single:

docker run -v $(pwd)/sound_rec/train_unlabled:/sound_rec/train_unlabled
audioproc record_single --device /dev/snd:/dev/snd

### for record on two microphones and convert to flack:

docker run -v $(pwd)/sound_rec/record_buffer:/sound_rec/record_buffer \
-v $(pwd)/sound_rec/flac:/sound_rec/flac audioproc record_two --device /dev/snd:/dev/snd

# crontab_tasks.txt

```
evocargo@evocargo-ubuntu-pi:~$ crontab -l
@reboot sleep 60 && python3 ~/audio_proc/commands.py "record_single"

*/15 * * * * python3 ~/audio_proc/connection_move_s3.py
```

# recording_errors.txt

```
20230205-003353: Error: 'RecordingFile' object has no attribute '_stream'
20230208-080808: couldn't find the right microphone
```
