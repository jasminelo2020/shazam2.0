from re import L
import sounddevice as sd
from scipy.io.wavfile import write
import streamlit as st
import librosa
import model 
import numpy as np
import matplotlib.pyplot as plt

def record(duration):
    fs = 22050  # Sample rate
    seconds = duration  # Duration of recording

    with st.spinner('Recording audio...'):
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)

def top_barchart(top_genres):
    names = list(top_genres.keys())
    values = list(top_genres.values())

    fig, ax = plt.subplots()
    plt.bar(range(len(top_genres)), values, tick_label=names)
    plt.show()
    return fig

def waveform_viz(y, onsets):
    fig, ax = plt.subplots()
    plt.plot(y[:onsets[40]])
    for onset in onsets[:40]:
        plt.axvline(x = onset, color = 'r', linestyle = 'dashed')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    return fig

def pitch_viz(pitch_array):
    plt.figure(figsize = (7, 4))

    fig, ax = plt.subplots()
    plt.imshow(pitch_array.T, aspect='auto')
    plt.colorbar()

    custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    custom_labels = ['C', 'C♯, D♭', 'D', 'D♯, E♭', 'E', 'F','F♯, G♭', 'G', 'G♯, A♭', 'A', 'A♯, B♭', 'B']
    plt.yticks(custom_ticks, custom_labels)
    plt.title('Pitch')
    plt.xlabel('Segment')
    return fig

def timbre_viz(y_segments):
    plt.figure(figsize = (7, 4))

    timbre = model.create_timbre_and_pitch('output.wav', 40)[0]

    fig, ax = plt.subplots()
    plt.imshow(timbre.T)
    plt.colorbar()
    plt.title('Timbre')
    plt.xlabel('Segment')
    return fig

st.write("""
# Shazam 2.0
Click the button below to find the genre of any song!
""")

if st.button("click here"):
    record(20)
    with st.spinner('Analyzing song contents...'):

        # model
        genres = model.predict_genre('output.wav', 40)

        # show top n genres classified
        n = 5

        # create tabs for each subtopic
        tab1, tab2, tab3, tab4 = st.tabs(["Top Genres", "Waveform", "Pitch", "Timbre"])
        
        # top n as barchart
        with tab1:
            st.header("Top Genres")
            total = 0
            keys = list(genres.keys())
            for i in range(n):
                total += float(genres[keys[i]])
            to_plot = {}
            for i in range(n):
                to_plot[keys[i]] = float(genres[keys[i]]) / total
            top_song = keys[0]

            st.write("""
            Was your song a(n) {} song?

            These are the top {} song genres as classified by our model. Check 
            out the other tabs to learn more!
            """.format(top_song, n))
            barchart_viz = top_barchart(to_plot)
            st.pyplot(fig=barchart_viz, clear_figure=None, use_container_width=True)

            # full results
            with st.expander("See full result"):
                st.write(genres)

        
        # waveform visualization
        with tab2:
            st.header("y Waveform")
            st.write("""
            The audio was segmented in this way, at each of the dotted red lines.
            This was done using Librosa's onset detection.

            Here is the visualization of the waveform of your song:
            """)
            y, sr = librosa.load('output.wav')
            onsets = librosa.onset.onset_detect(y = y[22050*3:-22050*3], sr = sr, units = 'samples')
            waveform_visualization = waveform_viz(y, onsets)
            st.pyplot(fig=waveform_visualization, clear_figure=None, use_container_width=True)

        # pitch 
        with tab3:
            st.header("What is pitch?")
            st.write("""
            The pitch vector has 12 elements per segment. Pitch cuts the spectrogram 
            into 12 bands called chroma features, where each band corresponds to 
            musical notes from C to B. The first element corresponds to C, 
            specifically it is the sum of the energy of all frequency bands 
            corresponding to C. A value close to 1 means that band was dominant in
            that sample.

            Here is the visualization of the pitch of your song:
            """)
            pitch = model.create_timbre_and_pitch('output.wav', 40)[-1]
            pitch_visualization = pitch_viz(pitch)
            st.pyplot(fig=pitch_visualization, clear_figure=None, use_container_width=True)
        
        # timbre
        with tab4:
            st.header("What is timbre?")
            st.write("""
            Timbre is a high level abstraction of audio data into 12 elements. 
            It is calculated for each segment of the song, and represents the 
            “quality” of musical notes, and can be used to distinguish different 
            types of instruments and voices. 
            Roughly speaking, the first elements in timbre correspond to 
            loudness of the note, the second to the brightness of the note, 
            and the third to the flatness of the note.

            Here is the visualization of the timbre of your song:
            """)
            timbre_visualization = timbre_viz(y)
            st.pyplot(fig=timbre_visualization, clear_figure=None, use_container_width=True)


