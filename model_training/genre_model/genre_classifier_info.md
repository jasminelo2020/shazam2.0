## Overview
We were set on training a deep neural network that would take in pitch and/or timbre as input and would classify genre.
We tested models that had no convolution layers, we tested models with convolution layers that only accepted timbre or 
pitch as input, and we also tested the number of segments that the model would take in. After this testing, we decided 
on a 40 segment model that had two convolution layers for timbre and pitch separately, which would then be combined and
processed through three dense layers. More detailed info can be found on our poster under the process section.

## Model
### Training
The code used for the training of the genre model is provided in `NN_with_log.py`. The model works on 40 segments, and
takes in an input of (batch_size, 1, 40, 12) for pitch and timbre respectively.
- Training Loss: 0.207295
- Validation Loss: 0.019888 
- N epochs: 180
- Final accuracy: 99%

### For The Future
There are so many different hyperparamters we did not get to tune. Some examples are kernel size, number of channels,
learning rate, and the list is practically never ending because this is a neural net. Changing the kernel size from the
default (3x3) to something else would be particularly interesting. Perhaps it would make sense to have the model look
further back and forward in time by expanding the kernel window in the time dimension.