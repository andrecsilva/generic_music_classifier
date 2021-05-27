#!/bin/env python
""" An active learning python script to classify your music library into moods"""
import pathlib
from functools import reduce
from itertools import compress
import random
import sys

import hashlib
import shelve

import tensorflow as tf
import pandas as pd

import audio_preprocessing as ap



def clean_data(files_data):
    files_data.dropna(axis=0,inplace=True)
    files_data = files_data[(files_data.label==0 | files_data.label==1)]
    return files_data

def get_key(file_path):
    with open(file_path,'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

def preprocess_or_retrieve(file_path,db):
    key = get_key(file_path)
    if key in db:
        return db[key]
    data = ap.preprocess(tf.constant(file_path))
    db[key] = data
    return data

# preprocess the latest labeled examples and append to the pickle file
def preprocess_training_data(train_file,db):
    """Returns a pandas dataframe containing the a mfcc spectrogram of the files"""
    print("Preprocessing training data...")

    #TODO clean data here...
    files_data = pd.read_csv(train_file, names=["music", "label"], sep="\t")

    mfcc_data = files_data.music.map(lambda x : \
            tf.expand_dims(preprocess_or_retrieve(x,db),axis=-1))
    mfcc_data = pd.DataFrame(mfcc_data).join(files_data.label)

    return mfcc_data

#def convert_to_tensor(mfcc_data):
#    """Convert the dataset of mfcc numpy arrays and convert it to a tensor dataset."""
#    #convert to numpy arrays in mfcc_data to tensors
#    mfcc_data = preprocess_training_data(train_file)
#    data = mfcc_data.music.map(lambda x: tf.expand_dims(tf.convert_to_tensor(x), -1))
#    data = pd.DataFrame(data).join(mfcc_data.label)
#    return data

def balance_data(data):
    """Makes the distributions of the classes as fair as possible."""

    samples_per_class = data.shape[0]
    for c in data.label.unique():
        samples_per_class = min(samples_per_class, data[data.label == c].shape[0])

    train_data = data.sample(0)
    for c in data.label.unique():
        train_data = train_data.append(data[data.label == c].sample(n=samples_per_class))
    return train_data


def split_into_datasets(train_data,batch_size):
    """Splits the training data into two Tensorflow datasets: trainig and validation."""
    labels = train_data.pop("label")

    # 80/20 split into train and validation databases
    train_size = 4 * train_data.shape[0] // 5
    #val_size = train_data.shape[0] - train_size
    # steps_per_epoch= train_size // batch_size
    # validation_steps= val_size // batch_size

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_data[:train_size].music.tolist(), labels[:train_size].tolist())
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (train_data[train_size:].music.tolist(), labels[train_size:].tolist())
    )

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    print(
        f"Train Size:{train_size}, Val Size:{train_data.shape[0] - train_size},\
                Data_Size = {train_data.shape[0]}"
    )
    return train_ds,val_ds


def get_model(train_ds):

    #grab the input_shape
    for s,_ in train_ds.take(1):
        input_shape = s.shape
    
    input_shape = input_shape[1:]

    #dimensions of the spectrogram
    size1 = int(input_shape[0])
    size2 = int(input_shape[1])

    # normalize the pixels of the spectogram
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    norm_layer.adapt(train_ds.map(lambda x, _: x))

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
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    # model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            256, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
        )
    )
    model.add(
        tf.keras.layers.Dense(
            128, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
        )
    )
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
    )
    return model

def save_model(model,train_file):
    model.path = pathlib.Path(train_file.parent / (train_file.stem + "_model"))
    if not model.path.exists():
        model.path.mkdir() 
    model.save(model.path)

#def plot():
#    plt.plot(history.history["binary_accuracy"], label="bin_acc")
#    plt.plot(history.history["val_binary_accuracy"], label="val_bin_acc")
#    plt.legend()
#    plt.show()

def predict(model,to_predict,db,batch_size=5):
    #files_test_ds = tf.data.Dataset.from_tensor_slices(to_predict)
    #test_ds = files_test_ds.map( \
    #    lambda x: tf.expand_dims(preprocess_or_retrieve(x.numpy(),db), axis=-1)\
    #)
    #test_ds = tf.data.Dataset.from_tensor_slices(\
    #       [preprocess_or_retrieve(m,db) for m in to_predict])
    #test_ds = tf.data.Dataset.from_tensor_slices([print_first(m,db) for m in to_predict])

    #TODO: change this to a pure tensorflow dataflow
    mask = []
    def generator():
        for m in to_predict:
            print(m)
            try:
                yield tf.expand_dims(preprocess_or_retrieve(m,db),axis=-1)
                mask.append(True)
            except:
                print('Error')
                mask.append(False)

    test_ds = tf.data.Dataset.from_generator(generator,output_signature=\
            tf.TensorSpec(shape=[None,None,1],dtype=tf.float32))

    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.batch(batch_size,drop_remainder=True)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    # calculate predictions
    print(f'Number of music to predict: {len(to_predict)}')
    test_pred = model.predict(test_ds, verbose=1)

    pred = reduce(lambda x, y: x + y, test_pred.tolist())
    pred = list(zip(compress(to_predict,mask), pred))

    return pred

# grab the half of the most confusing examples to be labelled
def append_predicitions(predictions):
    with open(train_file, "a") as f:
        for p in predictions:
            f.write(f"{p[0]}\n")

def generate_playlist(path,music):
    playlist_file = pathlib.Path(path)
    with open(playlist_file,"w") as f:
        for m in music:
            f.write(f"{m}\n")

if __name__ == '__main__':
    batch_size = 5
    seed = 133742
    random.seed(seed)

    if len(sys.argv)<3:
        print("Usage: active_learning.py <music_directory> <training_data_file>")
        print("<music_directory>: root directory that contains mp3 files.")
        print("<training_data_file>: name of the file that contains the labelled examples. "+ \
                "If the file does not exists, one will be created with a" + \
                "few examples to be labelled.")
        sys.exit()

    music_dir = pathlib.Path(sys.argv[1])
    train_file = pathlib.Path(sys.argv[2])

    db = shelve.open('preprocessed_data.db')

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

    train_data = balance_data(preprocess_training_data(train_file,db))
    # shuffle
    train_data = train_data.sample(frac=1)

    train_ds,val_ds = split_into_datasets(train_data,batch_size)
    model = get_model(train_ds)

    num_epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

    print(history.history)

    # get all music available at the library and remove the ones in the training data
    # all_music = set(music_dir.glob('**/*.mp3')).union(set(music_dir.glob('**/*.flac')))
    train_music = pd.read_csv(train_file, names=["music", "label"], sep="\t")
    all_music = set(music_dir.glob("**/*.mp3")).difference(
        set(map(pathlib.Path, train_music.music))
    )

    # sample music for testing
    sample_test = [str(p) for p in random.sample(all_music, 20)]
    pred = predict(model,sample_test,db,batch_size)

    # order predictions by confusion
    pred = sorted(pred, key=lambda x: abs(x[1]))

    for p in pred:
        print(p)

    append_predicitions(pred)

    playlist_file = pathlib.Path(train_file.parent / (train_file.stem + "_to_evaluate.m3u"))
    generate_playlist(playlist_file,[x for x,y in pred])
    save_model(model,train_file)
