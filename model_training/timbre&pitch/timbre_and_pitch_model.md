Timbre and pitch was the greatest challenge we faced through this entire project. The reason being that the company that
provided the API to retrieve timbre and pitch got bought out, meaning there was no way for us to generate pitch and
timbre features from raw audio data. This is a huge issue, since our whole project idea is that we want to create a 
music genre classifier that works just like Shazam: it needs to work on short audio recordings. 
Our genre model works very well (99.6% accuracy) if we can give it good timbre and pitch information; so the challenge
was figuring out how we could do this.

## Pitch
For pitch, we though we could use `librosa.chroma_stft`. This honestly mostly worked, as you can see in the comparison
of MSD pitch and librosa pitch on the poster. However, when we took the mean squared error between MSD pitch and librosa
pitch, it was quite high so we just decided to train a neural net to get us pitch.

## Timbre
Compared to pitch, timbre seemed like an even more daunting task. Our understanding was that it was the result of PCA
reduction on some high dimensional audio data. We reached out to a researcher named DAn Ellis who originally worked on the 
MSD for help. He told us that timbre are "PCA projections of the mel spectral envelopes of duration-normalized beat segments".
This gave us some direction, but we ultimately failed to recreate it in a mathematical fashion like we had hoped. We also
had to rely on a neural net for this.

## Models
### Training Data
**The problem**: we want to learn how to get accurate pitch and timbre from raw audio data, but the MSD does not include raw
audio. Maybe we could download the songs from youtube and use that as the raw audio? That wouldn't work either, because the 
segmentation between `librosa.onset_detection` and MSD is different. We don't even know what audio Echo Nest calculated 
timbre and pitch on--maybe they had a couple seconds of silence at the beginning, or at the end. There is no documentation
available.

**Solution**: DAn was a massive help in this regard too. Luckily, he had worked on a project to recreate raw audio from timbre 
and pitch. We just needed to use the output of DAn's code as input to the model, and the timbre and pitch as the output
of the model. `create_data.mlx` uses DAn's matlab code to regenerate audio from timbre and pitch, and puts those values
into a csv format so it is easily useable for training. We created a train set with 1000 songs, a validation set with 200
songs, and a test set with 1000 songs. Note that these songs are all from the MSD subset. You will need to download most
of the matlab files hosted on this [github repo](https://github.com/tbertinmahieux/MSongsDB/tree/master/MatlabSrc) for create_data.mlx to work. Thanks DAn.

### Pitch model
The code used for the training of the pitch model is provided in `pitch_NN_training.py`. The model works per segment,
and each segment is normalized to be 6606 samples. The input to this model is just the raw audio waveform data as an array
of (6606, 1). Our final pitch model had the following loss:
- Training Loss: 0.052985
- Validation Loss: 0.067605
- N epochs: 14

### Timbre model
The code used for the training of the timbre model is provided in `timbre_NN_training.py`. This model also works per
segment. After attempting to train in the exact same way as pitch by using the raw waveform of the audio, we discovered 
that the model simply was not doing well. The loss was not going down at around 900. Since timbre is related to the 
mel-spectral envelopes, we decided to make a mel spectrogram (using `librosa.feature.melspectrogram`) from the 6606 
samples and use that as input to the model. The resulting model was much better but still did not get the loss we were 
hoping for; the project deadline was closing in so we had to settle for this model.
- Training Loss: 460.495707
- Validation Loss: 334.6243776593889
- N epochs: 60