import numpy as np
import pandas as pd
import h5py
import zarr
import dask.array
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import glob

path = input("Enter path of HDF5s to concatenate:")
print("\nPath is: " + path)

pathnames = glob.glob(path + '*.hdf5')
print(f'\nCombining HDF5s...')
print(f'from {pathnames[0]}...')
print(f'to {pathnames[len(pathnames)-1]}...')

dest_path = input('Enter path of new combined HDF5:')
print("\nSave path is: " + dest_path)

# confirm the user wants to continue
confirm = input('Do you want to continue? [y(es) or no]')

# keep asking if invalid answers are given
while ((confirm not in ['yes', 'y', 'Yes', 'Y']) & (confirm!='no')):
    print('Answer invalid!')
    confirm = input('Do you want to continue? [y(es) or no]')

# end the script if users input "no"
if(confirm == 'no'):
    quit()

# collect timepoint one for total time elapsed
t1 = datetime.now()

# combine all data into one HDF5 file
with h5py.File(dest_path, 'w') as dest_hdf5:

    dest_hdf5.create_group('all_brain_presynapses')

    for i in tqdm(range(len(pathnames))):

        source = pathnames[i]
        source_hdf5 = h5py.File(source, 'r')

        for key in source_hdf5['brain_presynaptic_sites'].keys():
            source_hdf5.copy(f'brain_presynaptic_sites/{key}', dest_hdf5['all_brain_presynapses'])

        source_hdf5.close()

# collect timepoint two for total time elapsed
t2 = datetime.now()

elapsed = t2-t1
elapsed = f'{elapsed.total_seconds()//60} min, {(elapsed.total_seconds()%60)//1} sec'

print('\nFinished writing combined HDF5...')
print(f'Save location: {dest_path}')
print(f'Time elapsed: ' + elapsed)
