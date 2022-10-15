# %%
# load neuron IDs for all neurons with known neurotransmitter (NT)

import pymaid
from contools import Celltype_Analyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
rm = pymaid.CatmaidInstance(url, token, name, password)

# pull all cells with known neurotransmitter identities
# cholinergic, GABAergic, and glutamatergic
NT_annot = 'mw primary neurotransmitter'
known_NT = Celltype_Analyzer.get_skids_from_meta_annotation(NT_annot, split=True, return_celltypes=True)

# all unknown skids in the brain
all_known_NT_skids = [x for sublist in [celltype.skids for celltype in known_NT] for x in sublist]
unknown_skids = pymaid.get_skids_by_annotation('mw brain neurons') + pymaid.get_skids_by_annotation('mw brain accessory neurons')
unknown_skids = unknown_skids + Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities')

unknown_skids = np.setdiff1d(unknown_skids, all_known_NT_skids)

# remove namespacing from annotation names
for i in range(len(known_NT)): known_NT[i].name = known_NT[i].name.replace('mw ', '')

# %%
# identify locations of all presynaptic sites of each NT type

# load neuron morphologies
neurons_NT = [pymaid.get_neurons(celltype.skids) for celltype in known_NT]

# identify all presynaptic sites for each NT type
presyn_NT = [neurons.connectors[neurons.connectors.type==0] for neurons in neurons_NT]

# check to make sure each presynaptic site is unique
unique_check = [len(presyns.connector_id)==len(np.unique(presyns.connector_id)) for presyns in presyn_NT]
print(unique_check)

# %%
# pull out image data for each cube
import dask.array
import zarr

# load N5 dataset for Seymour L1
store = zarr.N5FSStore('http://zstore1.lmb.internal/srv/n5/seymour.n5') #url to seymour n5 volume
container = zarr.open(store, mode='r')
arr = container['volumes/raw/c0/s0']
d_arr = dask.array.from_zarr(arr)

# %%
# pull a test cube

cube_shape = np.array([551,551,550]) # in nm, note that 551 is divisible by 3.8 and 550 by 50

offset_nm = np.array([0, 0, 6050])
resolution_nm = np.array([3.8, 3.8, 50])

# example cube
test_node = presyn_NT[0].iloc[0]

#get world coordinates
world_coord = np.array(test_node[['x','y','z']],dtype='float')
print('World Co-ordinates:',world_coord)

#get voxel coordinates
voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
print('Voxel Co-ordinates:',voxel_coord)

cube_offsets = (cube_shape/2)//resolution_nm
cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

# add one to cube_j so offset is equal on both sides
sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

# convert to array of matrices with pixel intensity information
im = np.array(sliced)

# plot image stack
fig = plt.figure()
for i in range(0,11):
    plt.subplot(2,6,i+1)
    plt.imshow(im[i],cmap='gray')
    plt.axis('off')
fig.tight_layout(pad=0.5)
plt.show()

# %%
# pull all cubes around presynaptic sites associated with known NTs

cube_shape = np.array([551,551,550]) # in nm, note that 551 is divisible by 3.8 and 550 by 50
offset_nm = np.array([0, 0, 6050])
resolution_nm = np.array([3.8, 3.8, 50])

cubes_NT = []
for presyns in presyn_NT:
    cubes = []
    for i in presyns.index:
        node = presyns.loc[i]

        #get world coordinates
        world_coord = np.array(node[['x','y','z']],dtype='float')
        print('World Co-ordinates:',world_coord)

        #get voxel coordinates
        voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
        print('Voxel Co-ordinates:',voxel_coord)

        cube_offsets = (cube_shape/2)//resolution_nm
        cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
        cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

        # add one to cube_j so offset is equal on both sides
        sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

        cubes.append(sliced)

    cubes_NT.append(cubes)

# %%
# pull all cubes around unknown presynaptic sites associated with brain or brain accessory neurons

unknown_neurons = pymaid.get_neurons(unknown_skids)

# identify all presynaptic sites for each NT type
unknown_presyn = unknown_neurons.connectors[unknown_neurons.connectors.type==0]

cubes_unknown = []
for i in unknown_presyn.index:
    node = unknown_presyn.loc[i]

    #get world coordinates
    world_coord = np.array(node[['x','y','z']],dtype='float')
    print('World Co-ordinates:',world_coord)

    #get voxel coordinates
    voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
    print('Voxel Co-ordinates:',voxel_coord)

    cube_offsets = (cube_shape/2)//resolution_nm
    cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
    cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

    # add one to cube_j so offset is equal on both sides
    sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

    cubes_unknown.append(sliced)

# %%
# format data and save as .n5, for use in Jan Funke's synister


# %%
##################
# export data for 2D CNNs (take just three images per presynaptic site)
# try this as an alternative to synister from Jan Funke's group

# pull a test cube
cube_shape = np.array([551,551,150]) # in nm, note that 551 is divisible by 3.8 and 550 by 50

offset_nm = np.array([0, 0, 6050])
resolution_nm = np.array([3.8, 3.8, 50])

# example cube
test_node = presyn_NT[0].iloc[0]

#get world coordinates
world_coord = np.array(test_node[['x','y','z']],dtype='float')
print('World Co-ordinates:',world_coord)

#get voxel coordinates
voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
print('Voxel Co-ordinates:',voxel_coord)

cube_offsets = (cube_shape/2)//resolution_nm
cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

# add one to cube_j so offset is equal on both sides
sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

# convert to array of matrices with pixel intensity information
im = np.array(sliced)

# plot image stack
fig = plt.figure()
for i in range(0,3):
    plt.subplot(1,3,i+1)
    plt.imshow(im[i],cmap='gray')
    plt.axis('off')
fig.tight_layout(pad=0.5)
plt.show()

# %%
# pull all cubes around presynaptic sites associated with known NTs

cube_shape = np.array([551,551,150]) # in nm, note that 551 is divisible by 3.8 and 550 by 50
offset_nm = np.array([0, 0, 6050])
resolution_nm = np.array([3.8, 3.8, 50])

cubes_NT_2D = []
for presyns in presyn_NT:
    cubes = []
    for i in presyns.index:
        node = presyns.loc[i]

        #get world coordinates
        world_coord = np.array(node[['x','y','z']],dtype='float')
        print('World Co-ordinates:',world_coord)

        #get voxel coordinates
        voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
        print('Voxel Co-ordinates:',voxel_coord)

        cube_offsets = (cube_shape/2)//resolution_nm
        cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
        cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

        # add one to cube_j so offset is equal on both sides
        sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

        cubes.append(sliced)

    cubes_NT_2D.append(cubes)

# pull all cubes around unknown presynaptic sites associated with brain or brain accessory neurons

unknown_neurons = pymaid.get_neurons(unknown_skids)

# identify all presynaptic sites for each NT type
unknown_presyn = unknown_neurons.connectors[unknown_neurons.connectors.type==0]

cubes_unknown_2D = []
for i in unknown_presyn.index:
    node = unknown_presyn.loc[i]

    #get world coordinates
    world_coord = np.array(node[['x','y','z']],dtype='float')
    print('World Co-ordinates:',world_coord)

    #get voxel coordinates
    voxel_coord = np.array((world_coord-offset_nm)/resolution_nm,dtype='int')
    print('Voxel Co-ordinates:',voxel_coord)

    cube_offsets = (cube_shape/2)//resolution_nm
    cube_i = np.array(voxel_coord - cube_offsets, dtype='int')
    cube_j = np.array(voxel_coord + cube_offsets, dtype='int')

    # add one to cube_j so offset is equal on both sides
    sliced = d_arr[cube_i[2]:cube_j[2]+1, cube_i[1]:cube_j[1]+1, cube_i[0]:cube_j[0]+1]

    cubes_unknown_2D.append(sliced)

# %%
# convert to np.arrays

cubes_unknown_2D_arrays = [np.array(cube) for cube in cubes_unknown_2D]

cubes_NT_2D_arrays = []
for cube_NT in cubes_NT_2D:
    cubes_NT_2D_arrays.append([np.array(cube) for cube in cubes_unknown_2D])

# save in some format, run in CNN with VGG16, etc.