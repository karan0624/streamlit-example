import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import scipy.io.wavfile as wav
import librosa
import keras
from st_audiorec import st_audiorec


def get_audio_features(audio_path, sampling_rate):
    y, sample_rate = librosa.load(
        audio_path,
        res_type="kaiser_fast",
        duration=2.5,
        sr=sampling_rate * 2,
        offset=0.5,
    )
    sample_rate = np.array(sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13), axis=1)

    pitches, magnitudes = librosa.core.pitch.piptrack(y=y, sr=sample_rate)
    pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]
    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    chroma = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate), axis=1)

    return np.concatenate([mfccs, pitches, magnitudes, chroma])


def save_audio(audio_data, fs=44100):
    """Save wav audio data to a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, fs, audio_data)
        return f.name


def load_model():
    """Load the serialized model from model.json and update weights from karan.h5"""
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("karan.h5")
    return loaded_model


# Header
st.title("Emotion Detection from Speech")
st.write(
    "This is a simple demo of emotion detection from speech using a pre-trained model."
)
st.divider()

# Audio recording
st.subheader("Record your audio:")
# Record button
wav_audio_data = st_audiorec()
sample_rate = 44100  # default sample rate
st.divider()

# Process the recording
if wav_audio_data is not None:
    if st.button("Process Recording", type="primary", use_container_width=True):
        st.write("\n")
        # Store the prediction
        prediction = None
        with st.spinner("Processing..."):
            # load the model from disk
            model = load_model()
            # convert bytes to numpy int16 array
            np_audio_data = np.frombuffer(wav_audio_data, dtype=np.int16)
            # save audio data to a temporary file
            audio_file_name = save_audio(np_audio_data, fs=sample_rate)
            # convert audio data to features
            audio_features = get_audio_features(audio_file_name, sample_rate)
            # expand dimensions for model input
            audio_features = np.expand_dims(audio_features, axis=0)

            # predict emotion
            prediction = model.predict(audio_features)

        # detected_emotion is an array of probabilities for each emotion
        emotions = [
            "anger 😡",
            "disgust 😖",
            "fear 😰",
            "happy 😃",
            "neutral 😐",
            "sad 😔",
            "surprise 🫢",
        ]
        # diplay the result as a bar chart with all the emotions on the x-axis
        # and the probability of each emotion on the y-axis
        st.subheader("Emotion Probability")
        st.write("\n\n")
        st.bar_chart(
            pd.DataFrame({"Emotion": emotions, "Probability": prediction[0]}),
            x="Emotion",
            y="Probability",
            use_container_width=True,
        )

        # display details in expander
        with st.expander(
            ":blue[Expand to see details about the feature vector]", expanded=False
        ):
            st.markdown(
                (
                    "The feature vector is a concatenation of the following:  \n"
                    "  - [13 coefficients] __MFCCs__ _(a representation of the short-term power spectrum of a sound)_  \n"
                    "  - [20 coefficients] __Pitch__ _(a perceptual property that allows the ordering of sounds on a frequency-related scale)_ \n"
                    "  - [20 coefficients] __Magnitude__ _(a measure of the intensity of a sound)_  \n"
                    "  - [12 coefficients] __Chroma__  _(a representation of the spectral energy of a sound)_  \n"
                )
            )
            st.markdown(
                (
                    "For the current recording, the feature vector is:  \n"
                    "  - MFCC: :red[{}]  \n"
                    "  - Pitch: :blue[{}]  \n"
                    "  - Magnitude: :green[{}]  \n"
                    "  - Chroma: :orange[{}]  \n"
                ).format(
                    ", ".join([f"{x:.2f}" for x in audio_features[0][:13]]),
                    ", ".join([f"{x:.2f}" for x in audio_features[0][13:33]]),
                    ", ".join([f"{x:.2f}" for x in audio_features[0][33:53]]),
                    ", ".join([f"{x:.2f}" for x in audio_features[0][53:65]]),
                )
            )
    else:
        st.write(":blue[___Click the button to process the recording.___]")
else:
    st.markdown(":blue[___Record a snippet of your speech to detect emotion.___]")
st.divider()

st.markdown("_Karan Singh Chauhan - 2024_")
