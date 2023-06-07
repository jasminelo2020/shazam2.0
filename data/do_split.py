import os
import pandas as pd

dirs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
data = '../../../../public/formatted_csv'

output = '../../../../public/train_test'

log = '../../../../public/train_test/log.txt'

# create a dictionary from the train test labels data
# dictionary for O(1) lookup time!
# Test = 0, Train = 1
d = {}
with open('msd_tagtraum_cd2c_fixed_train2000.cls', 'r') as f:
    for line in f:
        l = line.strip().split('\t')
        if l[1] == 'TEST':
            d[l[0]] = 0
        else:
            d[l[0]] = 1
         

for dirr in dirs:
    path = os.path.join(data, dirr)
    for csv in os.listdir(path):
        print(f'working on {os.path.join(path, csv)}')
        try:
            df = pd.read_csv(os.path.join(path, csv), index_col = 0)
            # drop columns that are genre = World | New Age
            df = df.loc[~df['genre'].isin(['World', 'New'])]

            # create new column where 0 means test and 1 means train
            df['type'] = df['track_id'].apply(lambda x: d[x])

            df.loc[df['type'] == 0].to_csv(os.path.join(output, 'test', dirr, csv[:4] + 'test.csv'), index = False)
            df.loc[df['type'] == 1].to_csv(os.path.join(output, 'train', dirr, csv[:4] + 'train.csv'), index = False)
            print('done')
        except Exception as e:
            print('---error encountered, logging---')
            f = open(log, 'a')
            f.write(os.path.join(path, csv))
            f.write('\n')
            f.write(str(e))
            f.write('\n\n')