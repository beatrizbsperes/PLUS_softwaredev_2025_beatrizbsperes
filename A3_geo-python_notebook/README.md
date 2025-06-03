# LiDAR Point Cloud Processing and DEM with Python

A simple Python toolkit for processing LiDAR point cloud data, including reading, downsampling, ground classification and DEM generation.

## Overview

Here you'll find a Jupyter notebook demonstrating basic LiDAR processing workflows using Python geospatial libraries. There is also a data folder in whcih you can use samples of simple point clouds.

## Features

- **Point Cloud Reading**: Load LAS files using laspy
- **Data Downsampling**: Spatial binning for performance optimization
- **Ground Classification**: Height-based filtering with neighborhood analysis
- **DEM Generation**: Create Digital Elevation Models through spatial interpolation
- **GeoTIFF Export**: Save results for use in GIS applications
- **Visualization**: Generate plots and analysis visualizations

## Libraries Used

This project leverages several key Python geospatial libraries:

- **[laspy](https://laspy.readthedocs.io/en/latest/)**: Standard LAS/LAZ file I/O
- **[scipy](https://docs.scipy.org/doc/scipy/reference/spatial.html)**: Spatial operations and interpolation
- **[rasterio](https://rasterio.readthedocs.io/en/latest/)**: Raster operations and GeoTIFF export
- **[numpy](https://numpy.org/doc/stable/)**: Numerical computations
- **[matplotlib](https://matplotlib.org/stable/contents.html)**: Data visualization

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Setup Environment

```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate a3_lidar

# Launch Jupyter notebook
jupyter notebook
```

## Useful links

- **laspy Documentation**: https://laspy.readthedocs.io/en/latest/
- **scipy.spatial Documentation**: https://docs.scipy.org/doc/scipy/reference/spatial.html
- **rasterio Documentation**: https://rasterio.readthedocs.io/en/latest/
- **numpy Documentation**: https://numpy.org/doc/stable/
- **matplotlib Documentation**: https://matplotlib.org/stable/contents.html

---

**Course**: Practice: Software Development (Python)   
**Author**: Beatriz Peres  