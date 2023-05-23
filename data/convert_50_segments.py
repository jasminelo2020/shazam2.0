import ast
import pandas as pd
import numpy as np
import os
import json

def convert_format(fp, save_name):
    def explode_array(row):
        id = row['track_id']
        tim = row['timbre']
        pit = row['pitch']
        n = len(tim)
        tim = tim[:n//50*50]
        tim = tim.reshape(n//50, 50, 12)

        pit = pit[:n//50*50]
        pit = pit.reshape(n//50, 50, 12)
        ids = [id] * (n//50)

        return ids, pd.Series(tim.tolist()), pd.Series(pit.tolist())


    df = pd.read_csv(fp)

    # convert strings back into arrays
    df['timbre'] = df['timbre'].transform(lambda x: np.array([np.array(ast.literal_eval(y)) for y in ast.literal_eval(x)]))
    df['pitch'] = df['pitch'].transform(lambda x: np.array([np.array(ast.literal_eval(y)) for y in ast.literal_eval(x)]))

    df = df.drop(columns = ['Unnamed: 0'])

    tims = []
    pits = []
    ids = []

    for row in range(df.shape[0]):
        index, tim, pit = explode_array(df.iloc[row])
        ids.append(index)
        tims.append(tim)
        pits.append(pit)

    a = pd.concat([pd.DataFrame(pd.concat(tims)), pd.DataFrame(pd.concat(pits))], axis = 1)
    a.columns = ['timbre', 'pitch']
    # pd.concat([a, pd.Series(ids).explode()], axis = 1)
    a = a.reset_index().drop(columns=['index'])

    a['track_id'] = pd.Series(ids).explode().reset_index()[0]

    a = a.merge(df[['track_id', 'genre']], on = 'track_id')
    
    a.to_csv(save_name)
    
    
    
# loading in csv files
csv_data = '../../../public/csv_data'
formatted_csv = '../../../public/formatted_csv'
pattern = "*.csv"

dirs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# iterate through each parent alphabet directory
for dirr in dirs:
    # iterate through each 1st order child directory
    for f_name in os.listdir(os.path.join(csv_data, dirr)):
        if f_name[-3:] == 'csv':
            print('working on ' + f_name)
            convert_format(os.path.join(csv_data, dirr, f_name), os.path.join(formatted_csv, dirr, f'{f_name[:-4]}_formatted.csv'))

#         print(os.path.join(formatted_csv, dirr, f'{f_name[:-4]}_formatted.csv'))
            