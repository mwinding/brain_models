import pymaid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import zarr
import dask.array
from datetime import datetime
from tqdm import tqdm
import sys
from joblib import Parallel, delayed

# import os
# path = os.getenv('some_setup')
# sys.path.append(path)

from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

# neurons of interest
brain_neurons = pymaid.get_skids_by_annotation('mw brain neurons') + pymaid.get_skids_by_annotation('mw brain accessory neurons')
input_neurons = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs').name]
input_neurons = [x for sublist in input_neurons for x in sublist]
skids = brain_neurons + input_neurons

# identify locations of all presynaptic sites of each NT type

# load neuron morphologies
neurons = pymaid.get_neurons(skids)

# identify all presynaptic sites
presyn = neurons.connectors[neurons.connectors.type==0]

# check to make sure each presynaptic site is unique
unique_check = len(presyn.connector_id)==len(np.unique(presyn.connector_id))
if(unique_check):
    print('All presynaptic site connectors are unique')
else: 
    print('Some presynaptic connectors are not unique')

######
# load N5 dataset for Seymour L1
store = zarr.N5FSStore('http://zstore1.lmb.internal/srv/n5/seymour.n5') #url to seymour n5 volume
container = zarr.open(store, mode='r')
arr = container['volumes/raw/c0/s0']
d_arr = dask.array.from_zarr(arr)

print('Seymour .n5 loaded...')

######
# pull all cubes around presynaptic sites associated with known NTs
cube_shape = np.array([551,551,550]) # in nm, note that 551 is divisible by 3.8 and 550 by 50
offset_nm = np.array([0, 0, 6050])
resolution_nm = np.array([3.8, 3.8, 50])

cubes_meta_data = []
cubes = []
for i in tqdm(presyn.index):
    node = presyn.loc[i]

    #get world coordinates
    world_coord = np.array(node[['x','y','z']],dtype='float')
    #print('World Co-ordinates:',world_coord)

    #get voxel coordinates
    voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
    #print('Voxel Co-ordinates:',voxel_coord)

    cube_offsets = (cube_shape/2)//resolution_nm
    cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
    cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

    # add one to cube_j so offset is equal on both sides
    sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

    meta_data = [world_coord, voxel_coord, node.connector_id, node.node_id, node.neuron]
    cubes_meta_data.append(meta_data)
    cubes.append(sliced)

print('550nm cubes pulled...')

# format data and save as hdf5
def save_intermediate_hdf5(path, cubes, cubes_meta_data, i):
    with h5py.File(path.replace('.hdf5', f'_intermediate-{i}.hdf5'), 'a') as f:

        print('started saving .hdf5 training data...')
        f.attrs['date'] = datetime.today().strftime('%Y-%m-%d')
        f.attrs['readme'] = ''

        presyn_group = f.create_group('brain_presynaptic_sites')

        for i, idx in enumerate(tqdm(cubes_meta_data.index)):
            key = str(i).zfill(6) # fill to 6 digits (because in this case, no NT type has >999,999 examples)
            cube_meta = cubes_meta_data.loc[idx]

            ds = presyn_group.create_dataset(key, data=np.asarray(cubes[idx]))
            ds.attrs['connector_id'] = cube_meta.connector_id
            ds.attrs['node_id'] = cube_meta.node_id
            ds.attrs['skeleton_id'] = cube_meta.skid
            ds.attrs['connector_voxels_zyx'] = np.array(cube_meta.voxel_coord)
            ds.attrs['connector_project_zyx'] = np.array(cube_meta.world_coord)
            #ds.attrs['connector_offset_zyx'] = [5.5, 145.5, 145.5]

    f.close()
    print('intermediate HDF5 saved.')

batch_size = 1000

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

cubes_batches = list(divide_chunks(cubes, batch_size))
cubes_meta_batches = list(divide_chunks(cubes_meta_data, batch_size))
cubes_meta_batches = [pd.DataFrame(data=meta_data, columns=['world_coord', 'voxel_coord', 'connector_id', 'node_id', 'skid']) for meta_data in cubes_meta_batches]

print(f'Cubes divided into {len(cubes_batches)} batches')

path = sys.argv[1]
job = Parallel(n_jobs=-2)(delayed(save_intermediate_hdf5)(path, cubes_batches[i], cubes_meta_batches[i], i) for i in tqdm(range(0, len(cubes_batches))))
