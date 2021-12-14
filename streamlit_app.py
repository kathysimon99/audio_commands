#[reference](https://github.com/HimanshuKGP007/covid-detection)

import os
from datetime import datetime
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
#import soundfile as sf
import os
import librosa
import glob
from helper import create_spectrogram, read_audio, record, save_record, preprocess, get_dataframe,  scaler_transform

st.title('Audio Commands')

page = st.sidebar.selectbox('Select', ('About', 'Make a Prediction'))

if page == 'About':
    st.write(' My name is Kathy Simon and I am a Data Scientist')
    st.write('This model is created to predict an audio command. Take a look at the next page.')

if page == 'Make a Prediction':
    st.write('What audio command are you saying?')
    
    model = load_model('best_cnn.h5')

    if st.button(f"Click to Record"):
        record_state = st.text("Recording...")
        duration = 1  # seconds
        fs = 22050
        myrecording = record(duration, fs)

    if st.button(f'Classify'):
        path_myrecording = f"./samples/{filename}.wav"
        with st.spinner("Classifying the audio command"):
            retro = preprocess(path_myrecording)
            retro1 = get_dataframe(retro)
            retro2 = scaler_transform(retro1)

            prediction = model.predict(retro2)
        st.success("Classification completed")
        st.header("Test Results:")
        st.write({prediction})