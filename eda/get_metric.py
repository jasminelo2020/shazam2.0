# takes in the track_id and a getter function
# locates that file in the MillionSongSubset directory, and applies the function to the hdf5 file

import hdf5_getters as getters
import os
import glob

def get_metric(track_id, func):
    pattern = "*.h5"
    msp = "/Users/user/MillionSongSubset"
    for root, dirs, files in os.walk(msp):
        files = glob.glob(os.path.join(root, pattern))
        for f in files:
            if f[39 : 39 + 18] == track_id:
                h5 = getters.open_h5_file_read(f)
                return func(h5)
