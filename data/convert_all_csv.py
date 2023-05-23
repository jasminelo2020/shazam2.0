import hdf5_getters as getters
import numpy as np
import os
import glob
import pandas as pd


# loading in the genres

song_genres = {
    'id': [],
    'genre': []
}

with open('msd_tagtraum_cd2c_edited.cls', 'r') as file:
    for line in file:
        s = line.split()
        song_genres['id'].append(s[0])
        song_genres['genre'].append(s[1])
song_genres = pd.DataFrame(song_genres)


# loading in h5 files
msp = '../../../public/data'
msp_csv = '../../../public/csv_data'
pattern = "*.h5"


# warning: the code errors sometimes, so whenever the script terminated with an error, it would just
# start it again, but from the next base_dir. not sure why it errored and I know it's not a good
# solution but we were moving fast

# base dirs is the outermost directory
base_dirs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# dirs are the directories within the base dirs
dirs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# iterate through each parent alphabet directory
for dirr in base_dirs:
    # iterate through each 1st order child directory
    for subdir in dirs:
        infos = []
        print('loading ' + os.path.join(msp, dirr, subdir))
        # iterate through each 2nd order child directory
        for subsubdir in dirs:
            root = os.path.join(msp, dirr, subdir, subsubdir)
            # iterate through each file in 'data/dirr/subdir/subsubdir/'
            for f in os.listdir(root):
                song = []
                with getters.open_h5_file_read(os.path.join(root, f)) as h5:
                    song.append(getters.get_track_id(h5))
                    # song.append(getters.get_artist_name(h5))
                    # song.append(getters.get_title(h5))
                    song.append(getters.get_segments_timbre(h5))
                    # song.append(getters.get_segments_loudness_max(h5))
                    song.append(getters.get_segments_pitches(h5))
                    # song.append(getters.get_segments_confidence(h5))
                    infos.append(song)
        print(f'done getting data for {os.path.join(msp, dirr, subdir)}')
    
        # create a df with the data (would contain info for 'data/A/A' directory, for example)
        info_df = pd.DataFrame(data = infos, columns = ['track_id', 'timbre', 'pitch'])
        info_df['track_id'] = info_df['track_id'].transform(lambda x: x.decode())
        merged_df = info_df.merge(song_genres, how = 'inner', right_on = 'id', left_on = 'track_id').drop(columns = ['id'])

        # converts each 2d array into a string for csv conversion reasons
        # this for loop is the bane of my existence ahhh
        timbres = []
        pitches = []

        for i in range(merged_df.shape[0]):
            pitches.append([str(list(x)).replace('\n', '') for x in merged_df['pitch'][i]])
            timbres.append([str(list(x)).replace('\n', '') for x in merged_df['timbre'][i]])

        merged_df['pitch'] = pitches
        merged_df['timbre'] = timbres

        merged_df.to_csv(os.path.join(msp_csv, dirr) + f'/{dirr}-{subdir}.csv')
        print(os.path.join(msp_csv, dirr) + f'/{dirr}-{subdir}.csv' + ' saved')
        print('--------------------------------------------------------------')
    
    print('\n' + os.path.join(msp, dirr) + ' done!\n\n')
    

