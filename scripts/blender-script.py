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