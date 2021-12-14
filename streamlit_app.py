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
#from tensorflow.keras import layers
#from tensorflow.keras import models
#import soundfile as sf
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
    
    model = load_model('best_cnn.h5')

    def upload_and_save_wavfiles(save_dir: str) -> List[Path]:
        """ limited 200MB, you could increase by `streamlit run foo.py --server.maxUploadSize=1024` """
        uploaded_files = st.file_uploader("upload", type=['wav', 'mp3'], accept_multiple_files=True)
        save_paths = []
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                if uploaded_file.name.endswith('wav'):
                    audio = pydub.AudioSegment.from_wav(uploaded_file)
                    file_type = 'wav'
                elif uploaded_file.name.endswith('mp3'):
                    audio = pydub.AudioSegment.from_mp3(uploaded_file)
                    file_type = 'mp3'

                save_path = Path(save_dir) / uploaded_file.name
                save_paths.append(save_path)
                audio.export(save_path, format=file_type)
        return save_paths

    def display_wavfile(wavpath):
        audio_bytes = open(wavpath, 'rb').read()
        file_type = Path(wavpath).suffix
        st.audio(audio_bytes, format=f'audio/{file_type}', start_time=0)


    files = upload_and_save_wavfiles('temp')

    for wavpath in files:
        display_wavfile(wavpath)











    #new_file = st.file_uploader('Update .wav file here', type = 'wav')
    #new_audio, _ = tf.audio.decode_wav(new_file)
    #if st.button("Record"):
    #   record_state = st.text("Recording...")
    #    duration = 1  # seconds
    #   fs = 22050
    #    myrecording = record(duration, fs)

    if st.button('Classify'):
        with st.spinner("Classifying the audio command..."):
            spectrogram = get_spectrogram(new_audio)
            spectrogram_df = get_dataframe(spectrogram)
            
            prediction = model.predict(spectrogram_df)
        st.success("Classification completed")
        st.header("Test Results:")
        st.write({prediction})