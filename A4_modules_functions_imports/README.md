# Sentinel-2 Image Segmentation and Land Cover Classification unsing NDVI

A Python module and Jupyter Notebook workflow for object-based image analysis (OBIA) of Sentinel-2 satellite imagery. This toolkit provides tools for segmentation, NDVI computation and land cover classification of vegetation, water bodies and other surfaces. 

## Overview

This repository includes a Python script for processing Sentinel-2 scenes, along with a Jupyter Notebook demonstrating each function step-by-step. The goal is to showcase OBIA-based classification using spectral bands and segmentation.

You can use this workflow with your own Sentinel-2 images in GeoTIFF format.

## Features

- **Image Loading**: Read Sentinel-2 GeoTIFF images with selected RGB+NIR bands
- **Image Segmentation**: Superpixel segmentation using `skimage.slic`
- **NDVI Calculation**: Compute Normalized Difference Vegetation Index
- **Land Cover Classification**: Segment-wise classification of vegetation, water and other surface types
- **Visualization**: Display segmented regions and classification map
  

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Setup Environment

```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate a4_obia

# Launch Jupyter notebook
jupyter notebook
```

## Usage

- Use the Python script (`a4_obia.py`) to access individual functions in other workflows
- Use the accompanying notebook (`A4_OBIA_routine.ipynb`) to run a full pipeline on your image
- Make sure your image contains the expected Sentinel-2 bands in order: Red (B4), Green (B3), Blue (B2) and NIR (B8)
---

**Course**: Practice: Software Development (Python)   
**Author**: Beatriz Peres  
