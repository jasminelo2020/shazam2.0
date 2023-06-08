# shazam2.0
Music Genre Classifier Project for Data Science Student Society @ UCSD

## Introduction
Shazam is a service that can identify a song based on a short segment of a song. Our project aims to create a similar 
product which can determine the genre of a song, Shazam style. We utilized the Million Song dataset and various Python 
packages to create a CNN model that classifies the song genre of raw audio data. This GitHub repository contains our 
code from the process of creating our models and deliverable, a song genre classifier.

## Dataset
The Million Song Dataset (MSD) is a collection of audio features and metadata for 1 million songs published before 2012.
Some audio features were extracted using the Echo Nest API, an API that was provided by Echo Nest, now part of Spotify. 
We augmented this dataset with the Tagtrum MSD Genre Labels Dataset, since MSD does not come with genre labels. The 
combined dataset has around 200k rows. We lose around 4/5th of our dataset since many songs do not have a corresponding 
label in Tagtrum. MSD has many interesting metrics including danceability, loudness, and song hotness, but for our 
project, we are interested in segments, pitch, and timbre.

## Final Deliverable
Our deliverable is a genre classifier tool that we created using Streamlit. The website prompts you to record some
snippet of a song and the model will classify the genres, as well as showing you other metrics it calculated to arrive
at a genre classification.

## About This Repo
This repo contains the code that our group used to create the genre classifier. It will walk you through the exploratory
data analysis (EDA), data preprocessing, model training, and how we created our deliverable with streamlit. Each
section has some python, alongside a markdown file that walks you through the steps.

## Some final thoughts


## Contact
So Hirota
- Email: hirotaso92602@gmail.com
- Github: [soh09](https://github.com/soh09)

Sean Furhman
- Email: seantfuhrman@gmail.com
- Website: [seanfuhrman.com](https://seanfuhrman.com)

Jasmine Lo
- Email: j2lo@ucsd.edu
- Github: [jasminelo2020](https://github.com/jasminelo2020)

