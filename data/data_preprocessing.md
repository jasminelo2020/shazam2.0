I will explain the data cleaning process here.

#### 1. Make csvs
I made a csv file for each letter/letter directory, meaning I had A-A.csv to Z-Z.csv. You can do this with 
`convert_all_csv.py`. This script isn't perfect and it errors sometimes. I wrote how I handled errors in the script. I 
suspect it's something to do with the formatting of the hdf5 files. After running this script, you should have A-A.csv
to Z-Z.csv.


#### 2. Divide the songs into 50 segments
Once that's done, we somewhat arbitrarily decided that 50 segments would be a good input length for
the models that we were going to try to train. This is an important thing to note because I wrote code that iterates
through all the hdf5 files, gets timbre and pitch, and converts the 12 x n length array into 12 x 50 x m array. So 
basically the song gets exploded into m rows of 12 x 50 segments of timbre and pitch. 
You can run `convert_50_segments.py` to do this. The code is slow due to the fact that it has to convert timbre and 
pitch back to real lists (they are imposter lists in Step 1, meaning they are a list, where each element is a string 
that looks like a list).

```
Essentially, if we had a song with 500 segments, the format of the csv is:
----------------------------------------------------------------------------------------------------
timbre,pitch,track_id,genre
row 0: [[timbre 0], .... , [timbre 49]], [[pitch 0], .... , [pitch 49]], song_id_1, rock
    .
    .
    .
row 9: [[timbre 450], .... , [timbre 499]], [[pitch 450], .... , [pitch 499]], song_id_1, rock
    .
more songs....
    .
----------------------------------------------------------------------------------------------------
```

You can vary the number of segments per row by changing the part where I reshape the timbre and pitch arrays.

#### 3. Address the class imbalance
Perhaps you saw it in the poster, but this dataset has a pretty bad imbalance, so we need to address that somehow or
the model will just always predict Electronic. Luckily for us, tagtrum to the rescue again! They have a 
file called `msd_tagtrum_cd2c_fixed_train2000.cls` where all the trackIDs are annotated as TEST or TRAIN. `do_split.py`
just goes through all the csvs made in Step 2 and then divides them into a test and train folders. After this process,
you should have exactly 2000 songs for each genre, and we can now train using this for unbiased training. Note that the 
tagtrum annotations discarded all genres that did not have more than 2000 songs, like 'World' for example.

Finally, the data preprocessing is done. We can start training the model now.

