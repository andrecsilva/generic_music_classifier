# Generic Music Classifier

An active learning script to generate a binary music classifier written in Python with Tensorflow/Pandas/Librosa. Uses a simple 2D convolutional neural network over MFCC features extracted.

# Example of Usage

```bash
python active_learning <path_to_mp3_library> <training_examples.txt>
python playlist_generator.py <playlist_file> <model_folder> [directory | folder]+"
```

If `training_examples.txt` is an empty file, the script will sample 20 music from you library and write them there. You should label any music that fits your criteria with `1` and `0` otherwise. 

For example, if we want the model to detect metal music, then we should label any metal music with `1` and the rest with `0`.

Running the script will print some predictions with increasing order of reliability. These will be appended to `training_examples.txt` to be labelled. Additionally a playlist will be generated with the added examples.

The model and the preprocessed data will be saved when the script is finished.

Rinse and repeat.

When you're satisfied with the model accuracy, you can use `playlist_generator.py` to generate a playlist based on the model. You can adjust the cutoff for inclusion on the playlist on the file (default is 0.5).

Should work fairly well (~80%) as a (single) genre and "mood" classifier with around 100-200 labeled examples (with 50% split). 

# Why?

Why not? In particular it is quite useful to organize my music library.
