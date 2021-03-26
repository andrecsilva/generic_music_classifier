# Generic Music Classifier

An active learning script to generate a binary music classifier written in Python with Tensorflow/Pandas/Librosa. Uses a simple convolutional neural network.

# Example of Usage

```bash
./active_learning <path_to_mp3_library> <training_examples.txt>
```

If `training_examples.txt` is an empty file, the script will sample 20 music from you library and write them there. You should label any music that fits your criteria with `1` and `0` otherwise. 

For example, if we want the model to detect metal music, then we should label any metal music with `1` and the rest with `0`.

Running the script afterwards will print some predictions with increasing order of reliability. These will be appended to `training_examples` to be labelled.

Rinse and repeat.

Should work fairly well (~80%) as a (single) genre and "mood" classifier with around 100-200 labeled examples (with 50% split). 

# Why?

Why not? In particular it is quite useful to generate playlists as a tool to organize my music playlist.
