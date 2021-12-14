#[reference](https://github.com/HimanshuKGP007/covid-detection)

import os
from datetime import datetime
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
import streamlit as st
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import models
#import soundfile as sf
import os
import librosa
import glob
from helper import get_spectrogram, read_audio, record, get_dataframe

st.title('Audio Commands')

page = st.sidebar.selectbox('Select', ('About', 'Make a Prediction'))

if page == 'About':
    st.write(' My name is Kathy Simon and I am a Data Scientist')
    st.write('This model is created to predict an audio command. Take a look at the next page.')

if page == 'Make a Prediction':
    st.write('What audio command are you saying?')
    
    model = load_model('best_cnn.h5')

    audio = st.file_uploader('Update .wav file here', type = '.wav')

    #if st.button("Record"):
    #   record_state = st.text("Recording...")
    #    duration = 1  # seconds
    #   fs = 22050
    #    myrecording = record(duration, fs)

    #if st.button('Classify'):
    #    with st.spinner("Classifying the audio command..."):
    #        spectrogram = get_spectrogram(myrecording)
    #        spectrogram_df = get_dataframe(spectrogram)
            
    #        prediction = model.predict(spectrogram_df)
    #    st.success("Classification completed")
    #    st.header("Test Results:")
    #    st.write({prediction})