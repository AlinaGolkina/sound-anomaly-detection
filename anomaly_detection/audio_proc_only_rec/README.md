# Record process for docker

docker build . -t audioproc

### for record_single: 
docker run -v $(pwd)/sound_rec/train_unlabled:/sound_rec/train_unlabled audioproc record_single --device /dev/snd:/dev/snd            

### for record on two microphones and convert to flack:
docker run -v $(pwd)/sound_rec/record_buffer:/sound_rec/record_buffer \
-v $(pwd)/sound_rec/flac:/sound_rec/flac audioproc record_two --device /dev/snd:/dev/snd 
