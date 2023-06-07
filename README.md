# shazam2.0
Music Genre Classifier Project for DS3 @ UCSD

## Introduction
Shazam is a service that can identify a song based on a short segment of a song. Our project aims to create a similar product which can determine the genre of a song, Shazam style. We utilized the Million Song dataset and various Python packages to create a CNN model that classifies the song genre of raw audio data. This GitHub repository contains our code from the process of creating our models and deliverable, a song genre classifier.

## Dataset
The Million Song Dataset (MSD) is a collection of audio features and metadata for 1 million songs published before 2012. Some audio features were extracted using the Echo Nest API, an API that was provided by Echo Nest, now part of Spotify. We augmented this dataset with the Tagtrum MSD Genre Labels Dataset, since MSD does not come with genre labels. The combined dataset has around 200k rows. We lose around 4/5th of our dataset since many songs do not have a corresponding label in Tagtrum. MSD has many interesting metrics including danceability, loudness, and song hotness, but for our project, we are interested in segments, pitch, and timbre.

### Segments
- The MSD does not contain raw audio data. Instead, it contains numeric information for each segment. Each segment roughly represent one note, and comes with two metrics, duration and confidence. Each segment has a corresponding timbre and pitch.
- A segment duration is roughly 0.3 seconds.

### Pitch
- Pitch vector has 12 elements per segment. Pitch cuts the spectrogram into 12 bands called chroma features, where each band corresponds to musical notes from C to B. The first element corresponds to C, specifically it is the sum of the energy of all frequency bands corresponding to C.

### Timbre
- Timbre is a high level abstraction of audio data into 12 elements. It is calculated for each segment of the song, and represents the “quality” of musical notes, and can be used to distinguish different types of instruments and voices. 
  - Roughly speaking, the first elements in timbre correspond to loudness of the note, the second to the brightness of the note, and the third to the flatness of the note.

