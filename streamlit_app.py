#[reference](https://github.com/HimanshuKGP007/covid-detection)

import os
from datetime import datetime
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
import streamlit as st
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.keras import layers
#from tensorflow.keras import models
#import soundfile as sf
from scipy.io import wavfile as wav
import os
import librosa
import glob
import pydub
from helper import get_spectrogram, read_audio, record, get_dataframe

st.title('Audio Commands')

page = st.sidebar.selectbox('Select', ('About', 'Make a Prediction'))

if page == 'About':
    st.write(' My name is Kathy Simon and I am a Data Scientist')
    st.write('This model is created to predict an audio command. Take a look at the next page.')

if page == 'Make a Prediction':
    st.write('What audio command are you saying?')
    
    model = load_model('best_cnn.h5', compile = True)

    uploaded_file = st.file_uploader('Update .wav file here', type = 'wav')
    filename = './samples/' + uploaded_file.name
    st.write('Filename: ', filename)

    
 
    raw_audio = tf.io.read_file(filename)
    waveform, _ = tf.audio.decode_wav(raw_audio)

    #if st.button("Record"):
    #   record_state = st.text("Recording...")
    #    duration = 1  # seconds
    #   fs = 22050
    #    myrecording = record(duration, fs)

    if st.button('Classify'):
        with st.spinner("Classifying the audio command..."):
            input_len = 16000
            waveform = waveform[:input_len]

            zero_padding = tf.zeros([16000,1],
                dtype=tf.float32)
            st.write(tf.shape(waveform)[0], zero_padding)
            waveform = tf.cast(waveform, dtype=tf.float32)            
            equal_length = tf.concat([waveform, zero_padding], 0)
            spectrogram = tf.signal.stft(equal_length, frame_length=256, frame_step=128)
            spectrogram = tf.abs(spectrogram)
            spectrogram = spectrogram[..., tf.newaxis]
            st.write(spectrogram)
            test_audio = np.array(spectrogram)


            prediction = np.argmax(model.predict(test_audio), axis=1)
        st.success("Classification completed")
        st.header("Test Results:")
        st.write({prediction})