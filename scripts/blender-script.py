# Example script for CAVE/neuroglancer import to blender

import navis
import cloudvolume as cv

import navis.interfaces.blender as b3d

# This makes cloudvolume return navis objects
navis.patch_cloudvolume()

# Initialise the volume
vol = cv.CloudVolume("graphene://middleauth+https://data.proofreading.zetta.ai/segmentation/table/pg_fly_larva_aff_0_38_whole_v0", use_https=True)

# Initialise the handler
h = b3d.Handler()

# Fetch meshes/skeletons
meshes = vol.mesh.get_navis([648518346365681673])

# Skeletons (if provided by the datasource work similar):
# skels = vol.skeleton.get_navis([720575940603231916, 720575940605102694])

# Add to the Blender scene
h.add(meshes)

###########################################
# Example script for CATMAID import to blender

import pymaid as pymaid
from pymaid_creds import url, name, password, token
import navis.interfaces.blender as b3d
import numpy as np
import pymaid as pymaid

pymaid.CatmaidInstance(url, token, name, password)

h = b3d.Handler()

skids = pymaid.get_skids_by_annotation('mw MBONs')
neurons = pymaid.get_neurons(skids)
h.add(neurons)
h.neurons.bevel(0.02)
