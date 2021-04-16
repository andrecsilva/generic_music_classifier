# coding: utf-8
"""A mini-library to preprocess audio for tensorflow"""
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import numpy as np

#standard rate of samples
std_rate = 44100 //4

#pad/trucate to 5 mins of music at most
max_samples = std_rate * 2 * 60

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
def decode_audio(file_path):
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    return audio_tensor.to_tensor()

@tf.function(input_signature=(tf.TensorSpec(shape=[None,2], dtype=tf.float32),))
def multi_channel_to_mono(tensor):
    #nchannels = tf.cast(tensor.shape[1],dtype=tf.float32)
    #nchannels = tf.constant(tensor.shape[1])
    nchannels = tf.cast(tf.shape(tensor)[1],dtype=tf.float32)
    tensor = tf.reduce_sum(tensor,1)
    tensor = tf.math.divide(tensor,nchannels)
    return tensor

#import sys
#truncate/pad the tensor to a size of max_samples
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
def resize(tensor):
    #tf.print(tf.shape(tensor),output_stream=sys.stdout)

    #the music is looped as much as possible
    nsamples = tf.shape(tensor)[0]
    #repeated = ([tensor] * (max_samples // nsamples)) + [tensor[:max_samples % nsamples]]
    div = max_samples // nsamples
    rmd = max_samples % nsamples

    repeated = tf.repeat(tensor,div,axis=0)
    repeated = tf.concat([repeated,tensor[:rmd]],axis=0)
    
    return repeated

#returns a waveform tensor with size max_samples from a music file
@tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.string),))
def preprocess(file_path):
    #tf.print(file_path,output_stream=sys.stdout)
    #decodification
    tensor = tfio.audio.AudioIOTensor(file_path,dtype=tf.float32)
    
    #resampling
    rate = tf.cast(tensor.rate,dtype=tf.int64)
    tensor = tfio.audio.resample(tensor.to_tensor(),rate,std_rate)

    #colapse all channels into 1
    tensor = multi_channel_to_mono(tensor)

    #trim noise and resize
    p = tfio.experimental.audio.trim(tensor,axis=0,epsilon=0.1)
    tensor = tensor[p[0]:p[1]]
    tensor = resize(tensor)

    tensor = tf.numpy_function(get_mfcc,[tensor],tf.float32)
    #tensor = tf.numpy_function(get_melspectogram,[tensor],tf.float32)

    return tensor

def get_melspectogram(waveform):
    spectrogram = librosa.feature.melspectrogram(waveform,sr = std_rate,n_fft=2048,hop_length=1024);
    spectrogram = librosa.amplitude_to_db(spectrogram,ref=np.min)
    return np.abs(spectrogram)

def get_mfcc(waveform):
    mfcc = librosa.feature.mfcc(waveform,std_rate,n_mfcc=20,n_fft=2048,hop_length=1024);
    return mfcc

#The functions below are painfully slow for mp3 files due to librosa.load

#decodes, resample and converts to monochannel all at the same time
def librosa_decode_audio(file_path):
    return librosa.load(file_path,mono=True,sr=std_rate)[0]
    
#truncate/pad the tensor to a size of max_samples
def librosa_resize(tensor):
    tensor = tensor[:max_samples]
    pad = np.zeros(max(0,max_samples - t.shape[0]),dtype=np.float32)
    return np.concatenate([tensor,pad],axis=0)

#returns a waveform tensor with size max_samples from a music file
def librosa_preprocess(file_path):
    tensor = librosa_decode_audio(file_path)
    tensor = librosa_resize(tensor)
    return tensor

