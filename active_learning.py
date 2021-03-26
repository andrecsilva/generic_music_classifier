#!/bin/env python
""" An active learning python script to classify your music library into moods"""
import pathlib
from functools import reduce
import random
import sys

import audio_preprocessing as ap

import tensorflow as tf
import pandas as pd


seed = 133742
random.seed(seed)

# TODO save model and use it
# TODO small script to generate a .mp3 list of unlabelled music

music_dir = pathlib.Path(sys.argv[1])
train_file = pathlib.Path(sys.argv[2])
# music_dir = pathlib.Path('/media/Data/Mp3')
# train_file = pathlib.Path('calm.txt')
pickle_file = pathlib.Path(train_file.parent / (train_file.stem + ".pkl"))

# Checks if there's training data. If not, sample a small subset of music for labelling
if not train_file.exists():
    all_music = set(music_dir.glob("**/*.mp3"))
    sample = random.sample(all_music, 20)
    sample = set(sample)

    train_file.touch(exist_ok=True)
    with open(train_file, "w") as f:
        for s in sample:
            f.write(f"{str(s)}\n")

    print(f"Please label the music in {str(train_file)}, and execute this script again")
    sys.exit()

start = 0
mfcc_data = pd.DataFrame(columns=["music", "label"])
if pickle_file.exists():
    mfcc_data = pd.read_pickle(pickle_file)
    start = mfcc_data.shape[0]
files_data = pd.read_csv(train_file, names=["music", "label"], sep="\t")

# look at difference of rows in the txt and pkl files

# preprocess the latest labeled examples and append to the pickle file
print("Preprocessing training data...")
newest_data = files_data[start:].music.map(lambda x: ap.preprocess(x).numpy())
newest_data = pd.DataFrame(newest_data).join(files_data.label[start:])
mfcc_data = mfcc_data.append(newest_data)
mfcc_data.to_pickle(pickle_file)

# remove them from the rest of the set
# all_music = all_music.difference(sample)

# read the train data and shuffle
# raw_data = pd.read_csv(train_file,names=['music','label'],sep='\t')

data = mfcc_data.music.map(lambda x: tf.expand_dims(tf.convert_to_tensor(x), -1))
data = pd.DataFrame(data).join(mfcc_data.label)

del mfcc_data

# the data may be biased for a particular class, we make sure that the
# distribution is as fair as possible

samples_per_class = data.shape[0]
for c in data.label.unique():
    samples_per_class = min(samples_per_class, data[data.label == c].shape[0])

train_data = data.sample(0)
for c in data.label.unique():
    train_data = train_data.append(data[data.label == c].sample(n=samples_per_class))

# shuffle
train_data = train_data.sample(frac=1)

labels = train_data.pop("label")

# 80/20 split into train and validation databases
batch_size = 5
train_size = 4 * train_data.shape[0] // 5
val_size = train_data.shape[0] - train_size
# steps_per_epoch= train_size // batch_size
# validation_steps= val_size // batch_size

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_data[:train_size].music.tolist(), labels[:train_size].tolist())
)
val_ds = tf.data.Dataset.from_tensor_slices(
    (train_data[train_size:].music.tolist(), labels[train_size:].tolist())
)
# train_ds = tf.data.Dataset.from_tensor_slices((train_data[:train_size],labels[:train_size]))
# val_ds = tf.data.Dataset.from_tensor_slices((train_data[train_size:],labels[train_size:]))
#
# train_ds = train_ds.map(lambda x,y: (ap.get_spectogram(ap.preprocess(x)),y),\
#        num_parallel_calls=tf.data.AUTOTUNE)
# val_ds = val_ds.map(lambda x,y: (ap.get_spectogram(ap.preprocess(x)),y),\
#        num_parallel_calls=tf.data.AUTOTUNE)

# normalize the pixels of the spectogram
norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(train_ds.map(lambda x, _: x))

for spec, _ in train_ds.take(1):
    input_shape = spec.shape

size1 = int(input_shape[0])
size2 = int(input_shape[1])

num_epochs = 10

train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.batch(batch_size)
tin_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.batch(batch_size)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

print(
    f"Train Size:{train_size}, Val Size:{train_data.shape[0] - train_size},\
            Data_Size = {train_data.shape[0]}"
)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=input_shape))
# model.add(tf.keras.layers.experimental.preprocessing.Resizing(size1//8,size2//8))
model.add(norm_layer)
# model.add(tf.keras.layers.Conv2D(64,size//100,activation='relu'))
model.add(
    tf.keras.layers.Conv2D(
        64,
        (size1, size2),
        activation="relu",
        kernel_initializer=tf.keras.initializers.he_normal,
    )
)
# model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(
    tf.keras.layers.Dense(
        256, activation="relu", kernel_initializer=tf.keras.initializers.he_normal
    )
)
model.add(
    tf.keras.layers.Dense(
        128, activation="relu", kernel_initializer=tf.keras.initializers.he_normal
    )
)
model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
)

# history = model.fit(train_ds,validation_data=val_ds,epochs=num_epochs,\
#        steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

print(history.history)


#def plot():
#    plt.plot(history.history["binary_accuracy"], label="bin_acc")
#    plt.plot(history.history["val_binary_accuracy"], label="val_bin_acc")
#    plt.legend()
#    plt.show()


# get all music available at the library and remove the ones in the training data
# all_music = set(music_dir.glob('**/*.mp3')).union(set(music_dir.glob('**/*.flac')))
train_music = pd.read_csv(train_file, names=["music", "label"], sep="\t")
all_music = set(music_dir.glob("**/*.mp3")).difference(
    set(map(pathlib.Path, train_music.music))
)

# sample music for testing and gather it in to a dataset
sample_test = [str(p) for p in random.sample(all_music, 20)]
files_test_ds = tf.data.Dataset.from_tensor_slices(sample_test)
test_ds = files_test_ds.map(
    lambda x: tf.expand_dims(tf.convert_to_tensor(ap.preprocess(x)), axis=-1)
)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# calculate predictions
test_pred = model.predict(test_ds, verbose=1)

# order predictions by confusion
pred = reduce(lambda x, y: x + y, test_pred.tolist())
pred = list(zip(sample_test, pred))
pred = sorted(pred, key=lambda x: abs(x[1]))

for p in pred:
    print(p)

# grab the half of the most confusing examples to be labelled
def append_predicitions(predictions):
    with open(train_file, "a") as f:
        for p in predictions:
            f.write(f"{p[0]}\n")


append_predicitions(pred)
