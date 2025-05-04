# Sentinel-1 DEM Generation using Snappy and Snaphu

from snappy import ProductIO, GPF
from snappy import HashMap
import os

# Step 1: Load Sentinel-1 SLC Products
master = ProductIO.readProduct('/path/to/Master/manifest.safe')
slave = ProductIO.readProduct('/path/to/Slave/manifest.safe')

# Step 2: Apply Orbit Files
def apply_orbit(product):
    params = HashMap()
    params.put('Apply-Orbit-File', True)
    return GPF.createProduct('Apply-Orbit-File', params, product)

master_orbit = apply_orbit(master)
slave_orbit = apply_orbit(slave)

# Step 3: Back-Geocoding
params = HashMap()
params.put('demName', 'SRTM 1Sec HGT')
back_geocoded = GPF.createProduct('Back-Geocoding', params, [master_orbit, slave_orbit])

# Step 4: Coherence Estimation
params = HashMap()
params.put('cohWinAz', 10)
params.put('cohWinRg', 3)
params.put('squarePixel', False)
coherence = GPF.createProduct('Coherence', params, back_geocoded)

# Step 5: Interferogram Formation
params = HashMap()
params.put('subtractFlatEarthPhase', True)
params.put('includeCoherence', True)
interferogram = GPF.createProduct('Interferogram', params, coherence)

# Step 6: Goldstein Phase Filtering
params = HashMap()
params.put('alpha', 1.0)
filtered = GPF.createProduct('GoldsteinPhaseFiltering', params, interferogram)

# Step 7: Snaphu Export
export_folder = '/path/to/snaphu/export'
params = HashMap()
params.put('targetFolder', export_folder)
params.put('statCostMode', 'DEFO')
params.put('exportUnwrappedPhase', True)
params.put('tileExtensionPercent', 10)
params.put('numberOfLooks', 1)
params.put('rowOverlap', 200)
params.put('colOverlap', 200)
snaphu_export = GPF.createProduct('SnaphuExport', params, filtered)

ProductIO.writeProduct(snaphu_export, os.path.join(export_folder, 'snaphu-export'), 'BEAM-DIMAP')

# STEP 8: Run SNAPHU (done manually outside Python)
# Open terminal and run:
# cd /path/to/snaphu/export
# snaphu -s snaphu.conf unwPhase.img width height

# Step 9: Import Unwrapped Phase
wrapped = ProductIO.readProduct(os.path.join(export_folder, 'snaphu-export.dim'))
params = HashMap()
params.put('phaseFile', 'snaphu.unw')  # file created by snaphu
params.put('output', 'Phase')
params.put('copyMetadata', True)
snaphu_import = GPF.createProduct('SnaphuImport', params, wrapped)

# Step 10: Phase to Height
params = HashMap()
dem_product = GPF.createProduct('PhaseToHeight', params, snaphu_import)

# Step 11: Terrain Correction
params = HashMap()
params.put('demName', 'SRTM 1Sec HGT')
params.put('pixelSpacingInMeter', 10.0)
params.put('mapProjection', 'AUTO:42001')
corrected = GPF.createProduct('Terrain-Correction', params, dem_product)

# Step 12: Export final DEM as GeoTIFF
ProductIO.writeProduct(corrected, '/path/to/output/final_dem', 'GeoTIFF')