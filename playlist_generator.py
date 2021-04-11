#!/bin/env python
"""Generates a playlist based on the supplied model."""
import sys
import tensorflow as tf
import pathlib
from functools import reduce

from active_learning import predict,generate_playlist

if len(sys.argv)<3:
    print("Usage: playlist_generator.py <playlist_file> <model_folder> [directory | file]+")
    print("<playlist_file>: the name of the playlist file to be generated. ")
    print("<model_folder>: the folder containing the saved model to be used.")
    print("[directory|folder]: any sequence of mp3 files and folders to be analysed.")
    sys.exit()

cutoff = 0.4

playlist_file, model_path, *music = sys.argv[1:]

#grab all files in the directories 
music = [pathlib.Path(x) for x in music]
music = [m for l in map(lambda x : list(x.glob('**/*.mp3')) if x.is_dir() else [x],music) for m in l]

model = tf.keras.models.load_model(model_path)

#predict the scores
pred = predict(model,[str(p) for p in music])
#pred = sorted(pred,key=lambda x: x[1],reverse=True)

#generate playlist
playlist_file = pathlib.Path(playlist_file)
generate_playlist(playlist_file,[x for x,y in pred if y>=cutoff])
print(f'playlist generated at {playlist_file.absolute()}')

#statistics about the rejected files
rejected = [(x,y) for x,y in pred if y<cutoff];
rejected = sorted(rejected,key=lambda x:x[1])

print(f'The following were rejected dues to low score (<{cutoff}):')
for p in rejected:
    print(p)

print(f'Rejected {len(rejected)} out of {len(music)}.')
