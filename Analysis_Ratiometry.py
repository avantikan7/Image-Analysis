import os
import numpy as np
import matplotlib.pyplot as plt
import csv  # Add import for CSV module
from skimage import io, exposure, filters, measure, morphology, img_as_float
from skimage.filters import threshold_local

def load_channel_images(folder_path, channel_suffix):
    # Load images from a folder with a specific suffix for each channel
    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(f"_{channel_suffix}.tif")]
    )
    image_stack = [io.imread(img) for img in image_files]
    return np.array(image_stack)

def z_projection(image_stack):
    # Maximum intensity projection along the Z-axis
    return np.max(image_stack, axis=0)

def adaptive_threshold(image, block_size=35, offset=10):
    # Adaptive thresholding for improved bead segmentation
    thresh = threshold_local(image, block_size=block_size, offset=offset)
    binary = image > thresh
    return binary

def analyze_particles(binary_image, intensity_image):
    # Label particles and analyze their properties
    labeled_image = morphology.label(binary_image)
    properties = measure.regionprops(labeled_image, intensity_image)
    
    # Collect data for each bead
    results = []
    for prop in properties:
        if prop.area > 5:  # Filter small noise particles (adjust as needed)
            results.append({
                "Area": prop.area,
                "Mean Intensity": prop.mean_intensity,
                "Centroid": prop.centroid
            })
    return results

# Specify the folder where all images are stored
image_folder = "/Users/avantikan/Downloads/analysis"

# Load image stacks for each channel based on suffix
green_stack = load_channel_images(image_folder, "g")
orange_stack = load_channel_images(image_folder, "o")
red_stack = load_channel_images(image_folder, "r")

# Z-projection for each channel
green_proj = z_projection(green_stack)
orange_proj = z_projection(orange_stack)
red_proj = z_projection(red_stack)

# Enhance contrast using adaptive histogram equalization
green_proj = exposure.equalize_adapthist(img_as_float(green_proj))
orange_proj = exposure.equalize_adapthist(img_as_float(orange_proj))
red_proj = exposure.equalize_adapthist(img_as_float(red_proj))

# Adaptive thresholding
green_thresh = adaptive_threshold(green_proj)
orange_thresh = adaptive_threshold(orange_proj)
red_thresh = adaptive_threshold(red_proj)

# Analyze particles in each channel
green_particles = analyze_particles(green_thresh, green_proj)
orange_particles = analyze_particles(orange_thresh, orange_proj)
red_particles = analyze_particles(red_thresh, red_proj)

# Calculate ratios and print results
print("Green Channel Particles:", green_particles)
print("Orange Channel Particles:", orange_particles)
print("Red Channel Particles:", red_particles)

# Calculate ratiometric data for each corresponding bead (assuming similar bead centroids across channels)
ratios = []
for i in range(min(len(green_particles), len(orange_particles), len(red_particles))):
    ratio_GR = green_particles[i]['Mean Intensity'] / red_particles[i]['Mean Intensity']
    ratio_OR = orange_particles[i]['Mean Intensity'] / red_particles[i]['Mean Intensity']
    ratios.append({
        "Bead Index": i,
        "G/R Ratio": ratio_GR,
        "O/R Ratio": ratio_OR
    })

print("Ratiometric Data:", ratios)

# Save ratiometric data to a CSV file
csv_file = "ratiometric_data.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Bead Index", "G/R Ratio", "O/R Ratio"])
    writer.writeheader()
    writer.writerows(ratios)

print(f"Ratiometric data saved to {csv_file}")
