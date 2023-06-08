You can see some of the EDA we did on the poster, as we put some graphs for timbre and pitch aggregated by genre. The
notebooks we have included here are very much a work in progress and some of it might not work. EDA is kind of a process
you do yourself to discover trends in data and everybody has different ways of doing it. Some of the ways I did it is 
included as the notebook, but I encourage you do your own exploration.
Some Notes:
- The `hdf5_getters.py` file is provided by MSD, and it is the way you access metrics from .h5 files. Each song is its
own .h5 file, and the name of the file corresponds to its trackID.
- This EDA was done on the MSD subset (around a gigabyte in size). It is available on the MSD website as a direct download link (unlike the entire dataset). This way you can get your hands dirty with the actual dataset, but in
an accessible manner.
- `get_metric.py` is useful for getting a certain metric about a song. Like if you wanted to figure out the song title
of a song with TrackID of TRARREF128F422FD96, pass the TrackID and the getters.get_title function.