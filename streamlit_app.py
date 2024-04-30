import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import base64
import numpy as np
import librosa 
import pandas as pd 
from scipy.io.wavfile import write
from keras.models import model_from_json

def get_audio_features(audio_path,sampling_rate):
    X, sample_rate = librosa.load(audio_path ,res_type='kaiser_fast',duration=2.5,sr=sampling_rate*2,offset=0.5)
    sample_rate = np.array(sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=1)

    pitches = np.trim_zeros(np.mean(pitches,axis=1))[:20]

    magnitudes = np.trim_zeros(np.mean(magnitudes,axis=1))[:20]

    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate),axis=1)
    
    return [mfccs, pitches, magnitudes, C]
    
    

def get_features_dataframe(dataframe, sampling_rate):
    labels = pd.DataFrame(dataframe['label'])
    
    features  = pd.DataFrame(columns=['mfcc','pitches','magnitudes','C'])
    for index, audio_path in enumerate(dataframe['path']):
        features.loc[index] = get_audio_features(audio_path, sampling_rate)
    
    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)
    
    combined_features = pd.concat([mfcc,pit,mag,C],axis=1,ignore_index=True)

    return combined_features, labels 

def save_audio(audio_data, fs=44100):
    """Convert base64 audio to a .wav file and save it."""
    audio_bytes = base64.b64decode(audio_data.split(',')[1])
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    with open(temp_file.name, 'wb') as f:
        f.write(audio_bytes)
    return temp_file.name

from keras.models import model_from_json
def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/Users/apple/Desktop/software project /Trained_Models/karan.h5")
    print("Loaded model from disk")
    return loaded_model

def load_and_predict(audio_file, model):
    """Load audio, preprocess, and predict emotion."""
    # Load and preprocess the audio
    sampling_rate = 44100  # Adjust as needed
    features = get_audio_features(audio_file, sampling_rate)
    # Predict emotion
    emotion = model.predict(features)
    return emotion

# Streamlit interface
st.title("Emotion Detection from Speech")
st.write("This app detects your emotion based on a short speech recording.")

# Record button
audio_data = audio_recorder(recording_time=5000, st_key="recorder")  # 5000 ms = 5 seconds

if audio_data is not None:
    if st.button("Process Recording"):
        audio_file = save_audio(audio_data)
        emotion = load_and_predict(audio_file)
        st.write(f"Detected Emotion: {emotion}")
