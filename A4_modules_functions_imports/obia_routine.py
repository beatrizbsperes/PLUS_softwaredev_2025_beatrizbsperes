#!/usr/bin/env a4_obia
"""
Sentinel-2 Satellite Image Analysis and Land Cover Classification

This module provides functionality for analyzing Sentinel-2 satellite imagery using object-based
image analysis (OBIA) techniques, such as segmentation and later classification. 
It specifically works with Sentinel-2 bands:
- B2: Blue (490nm)
- B3: Green (560nm) 
- B4: Red (665nm)
- B8: NIR (842nm)

The module includes functions for loading Sentinel-2 images, performing image segmentation,
and classifying land cover types including vegetation, water bodies and other surface types.


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import segmentation, measure, filters
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.plot import show


def load_image(image_path, bands=['B4', 'B3', 'B2', 'B8']):
    """
    Load and read a Sentinel-2 satellite image from file.
    
    This function loads Sentinel-2 multi-spectral satellite imagery using rasterio,
    specifically designed for Sentinel-2 band structure with default RGBN bands.
    
    Parameters
    ----------
    image_path : str
        Path to the Sentinel-2 image file (GeoTIFF format recommended)
    bands : list of str, default=['B4', 'B3', 'B2', 'B8']
        List of Sentinel-2 band names to load (Red, Green, Blue, NIR)
        Default order provides RGBN for proper processing
    
    Returns
    -------
    image_array : numpy.ndarray
        3D array with shape (height, width, 4) containing RGBN data
    profile : dict
        Rasterio profile containing metadata (CRS, transform, etc.)
    
    Examples
    --------
    >>> # Load default RGBN bands from Sentinel-2 image
    >>> image, profile = load_sentinel2_image('S2_scene.tif')
    >>> print(f"Image shape: {image.shape}")
    """
    with rasterio.open(image_path) as src:
        profile = src.profile
        
        # For Sentinel-2, we expect 4 bands in RGBN order
        if len(bands) != 4:
            raise ValueError("Sentinel-2 processing requires exactly 4 bands: [Red, Green, Blue, NIR]")
        
        # Read all 4 bands (assuming they are in order B4, B3, B2, B8)
        image_array = src.read([1, 2, 3, 4])  # Read bands 1-4 from file
        # Transpose to (height, width, bands) format
        image_array = np.transpose(image_array, (1, 2, 0))
        
        # Normalize to 0-1 range for Sentinel-2 (typically 16-bit data)
        if image_array.dtype in [np.uint16, np.uint8]:
            # Store original dtype before conversion
            original_dtype = image_array.dtype
            image_array = image_array.astype(np.float32)
            if image_array.max() > 1.0:
                # Sentinel-2 Level-2A products are typically 0-10000 range
                if image_array.max() > 10000:
                    image_array = image_array / np.iinfo(original_dtype).max
                else:
                    image_array = image_array / 10000.0
        
    print(f"Loaded Sentinel-2 image with bands: {bands}")
    print(f"Image shape: {image_array.shape}")
    return image_array, profile
    
def image_segmentation(image, n_segments=1000, compactness=10, sigma=1):
    """
    This function segments the Sentinel-2 image into homogeneous regions
    that can be used for object-based analysis. The segmentation creates
    superpixels that respect image boundaries and spectral similarity.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input Sentinel-2 image with shape (height, width, 4) for RGBN
    n_segments : int, default=1000
        Approximate number of segments to generate
    compactness : float, default=10
        Balance between color proximity and space proximity.
        Higher values give more weight to space proximity (more compact segments)
    sigma : float, default=1
        Width of Gaussian smoothing kernel for preprocessing
    
    Returns
    -------
    segments : numpy.ndarray
        2D array with same height/width as input, containing segment labels
    n_segments_actual : int
        Actual number of segments created
    
    Examples
    --------
    >>> segments, n_segs = image_segmentation(image, n_segments=500)
    >>> print(f"Created {n_segs} segments")
    """
    # Use RGB bands (first 3 bands) for segmentation
    rgb_image = image[:, :, :3]
    
    # Apply Gaussian smoothing to reduce noise
    if sigma > 0:
        for i in range(rgb_image.shape[2]):
            rgb_image[:, :, i] = filters.gaussian(rgb_image[:, :, i], sigma=sigma)
    
    # Perform segmentation
    segments = segmentation.slic(
        rgb_image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        channel_axis=2
    )
    
    # Get actual number of segments
    n_segments_actual = len(np.unique(segments))
    
    print(f"Segmentation complete: {n_segments_actual} segments created")
    
    return segments, n_segments_actual

def calculate_ndvi(image, red_band_idx=0, nir_band_idx=3):
    """
    Calculate Normalized Difference Vegetation Index (NDVI) for Sentinel-2.
    
    NDVI = (NIR - Red) / (NIR + Red)
    For Sentinel-2: NDVI = (B8 - B4) / (B8 + B4)
    
    NDVI values typically range from -1 to 1:
    - Values > 0.3: Dense vegetation
    - Values 0.1-0.3: Sparse vegetation
    - Values < 0.1: Non-vegetation (water, urban, bare soil)
    
    Parameters
    ----------
    image : numpy.ndarray
        Input Sentinel-2 image with shape (height, width, 4) RGBN
    red_band_idx : int, default=0
        Index of the red band (B4) in the image array
    nir_band_idx : int, default=3
        Index of the near-infrared (B8) band in the image array
    
    Returns
    -------
    ndvi : numpy.ndarray
        2D array with NDVI values ranging from -1 to 1
    
    Examples
    --------
    >>> ndvi = calculate_ndvi(image)  # Uses default B4 and B8
    >>> vegetation_mask = ndvi > 0.3
    """
    # Check if we have the required bands
    if image.shape[2] <= max(red_band_idx, nir_band_idx):
        raise ValueError(f"Image has {image.shape[2]} bands, but trying to access band {max(red_band_idx, nir_band_idx)}")

    # Extract red (B4) and NIR (B8) bands
    red = image[:, :, red_band_idx].astype(np.float32)
    nir = image[:, :, nir_band_idx].astype(np.float32)
    
    # Calculate NDVI with small epsilon to avoid division by zero
    denominator = nir + red + 1e-8
    ndvi = (nir - red) / denominator
    
    # Clip values to valid NDVI range [-1, 1]
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi

def classify_land_cover(image, segments, vegetation_threshold=0.3, water_threshold=0.15, 
                       red_band_idx=0, nir_band_idx=3, use_ndvi=True):
    """
    Classify land cover into vegetation, water and other categories using NDVI.
    
    This function performs land cover classification for Sentinel-2 data based on NDVI values
    and mean pixel reflectance within each segment. Optimized for Sentinel-2 RGBN bands.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input Sentinel-2 image with shape (height, width, 4) RGBN
    segments : numpy.ndarray
        Segmentation labels from image_segmentation()
    vegetation_threshold : float, default=0.3
        Threshold for vegetation classification (NDVI)
    water_threshold : float, default=0.15
        Threshold for water classification (mean reflectance)
    red_band_idx : int, default=0
        Index of the red band (B4) in the image
    nir_band_idx : int, default=3
        Index of the NIR band (B8) in the image
    use_ndvi : bool, default=True
        Whether to use NDVI calculation (should always be True for Sentinel-2)
    
    Returns
    -------
    classification : numpy.ndarray
        2D array with classification labels:
        0 = Other/Unclassified, 1 = Vegetation, 2 = Water
    class_stats : dict
        Dictionary with statistics about each class
    ndvi_map : numpy.ndarray
        2D array with NDVI values
    
    Examples
    --------
    >>> classification, stats, ndvi = classify_land_cover(image, segments)
    >>> print(f"Vegetation covers {stats['vegetation_percent']:.1f}% of the image")
    """
    classification = np.zeros_like(segments, dtype=np.uint8)
    
    # Calculate NDVI using Sentinel-2 bands
    ndvi_map = calculate_ndvi(image, red_band_idx, nir_band_idx)
    print(f"Using NDVI with Red band {red_band_idx} (B4) and NIR band {nir_band_idx} (B8)")

    # Get unique segment labels
    segment_labels = np.unique(segments)
    
    # Initialize counters
    vegetation_pixels = 0
    water_pixels = 0
    other_pixels = 0
    
    # Store NDVI statistics for analysis
    ndvi_stats = []
    
    for label in segment_labels:
        # Create mask for current segment
        mask = segments == label
        
        # Extract pixel values for this segment
        segment_pixels = image[mask]
        segment_ndvi = ndvi_map[mask]
        
        # Calculate mean values
        mean_values = np.mean(segment_pixels, axis=0)
        mean_ndvi = np.mean(segment_ndvi)
        
        # Mean reflectance for water detection (average across all bands)
        mean_reflectance = np.mean(mean_values)
        
        # Store statistics
        ndvi_stats.append({
            'segment': label,
            'mean_ndvi': mean_ndvi,
            'mean_reflectance': mean_reflectance,
            'pixel_count': np.sum(mask)
        })
        
        # Classify based on thresholds
        if mean_ndvi > vegetation_threshold:
            classification[mask] = 1  # Vegetation
            vegetation_pixels += np.sum(mask)
        elif mean_reflectance < water_threshold:
            classification[mask] = 2  # Water
            water_pixels += np.sum(mask)
        else:
            classification[mask] = 0  # Other
            other_pixels += np.sum(mask)
    
    # Calculate statistics
    total_pixels = image.shape[0] * image.shape[1]
    class_stats = {
        'vegetation_pixels': vegetation_pixels,
        'water_pixels': water_pixels,
        'other_pixels': other_pixels,
        'vegetation_percent': (vegetation_pixels / total_pixels) * 100,
        'water_percent': (water_pixels / total_pixels) * 100,
        'other_percent': (other_pixels / total_pixels) * 100,
        'total_segments': len(segment_labels),
        'mean_ndvi': np.mean(ndvi_map),
        'max_ndvi': np.max(ndvi_map),
        'min_ndvi': np.min(ndvi_map),
        'ndvi_std': np.std(ndvi_map),
        'segment_stats': ndvi_stats
    }
    
    print(f"Classification complete:")
    print(f"  Vegetation: {class_stats['vegetation_percent']:.1f}%")
    print(f"  Water: {class_stats['water_percent']:.1f}%")
    print(f"  Other: {class_stats['other_percent']:.1f}%")
    print(f"  NDVI range: {class_stats['min_ndvi']:.3f} to {class_stats['max_ndvi']:.3f}")
    print(f"  Mean NDVI: {class_stats['mean_ndvi']:.3f}")
    
    return classification, class_stats, ndvi_map

def visualization(image, segments, classification, figsize=(15, 10)):
    """
    Visualize Sentinel-2 image analysis results.
    
    This function creates a comprehensive visualization showing the original
    Sentinel-2 image, segmentation results and land cover classification
    in a multi-panel figure.
    
    Parameters
    ----------
    image : numpy.ndarray
        Original Sentinel-2 image (RGBN)
    segments : numpy.ndarray
        Segmentation labels
    classification : numpy.ndarray
        Land cover classification results (0=Other, 1=Vegetation, 2=Water)
    figsize : tuple, default=(15, 10)
        Figure size in inches (width, height)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    axes : numpy.ndarray
        Array of subplot axes
    
    Examples
    --------
    >>> fig, axes = visualization(image, segments, classification)
    >>> plt.savefig('sentinel2_analysis_results.png', dpi=300, bbox_inches='tight')
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Sentinel-2 Image Analysis Results', fontsize=16, fontweight='bold')
    
    # Original image (RGB composite from B4, B3, B2)
    rgb_image = image[:, :, :3]
    
    # Improved normalization - stretch each band individually using percentiles
    rgb_enhanced = np.zeros_like(rgb_image)
    for i in range(3):
        band = rgb_image[:, :, i]
        # Use 2nd and 98th percentiles for better contrast
        p_low, p_high = np.percentile(band, (2, 98))
        rgb_enhanced[:, :, i] = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    
    axes[0, 0].imshow(rgb_enhanced)
    axes[0, 0].set_title('Original Sentinel-2 Image (RGB: B4,B3,B2)')
    axes[0, 0].axis('off')
    
    # Segmentation boundaries
    axes[0, 1].imshow(rgb_enhanced)
    axes[0, 1].imshow(segmentation.mark_boundaries(rgb_enhanced, segments, color=(1, 1, 0), outline_color=(1, 1, 0)), alpha=0.8)
    axes[0, 1].set_title(f'Image Segmentation ({len(np.unique(segments))} segments)')
    axes[0, 1].axis('off')
    
    # Classification result
    class_colors = ['gray', 'green', 'blue']  # Other, Vegetation, Water
    class_cmap = ListedColormap(class_colors)
    
    im = axes[1, 0].imshow(classification, cmap=class_cmap, vmin=0, vmax=2)
    axes[1, 0].set_title('Land Cover Classification')
    axes[1, 0].axis('off')
    
    # Add colorbar for classification
    cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.6)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Other', 'Vegetation', 'Water'])
    
    # Classification overlay on original image
    axes[1, 1].imshow(rgb_enhanced)
    
    # Create colored overlay for each class
    overlay = np.zeros_like(rgb_enhanced)
    overlay[classification == 1] = [0, 1, 0]  # Green for vegetation
    overlay[classification == 2] = [0, 0, 1]  # Blue for water
    
    axes[1, 1].imshow(overlay, alpha=0.2)
    axes[1, 1].set_title('Classification Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig, axes

