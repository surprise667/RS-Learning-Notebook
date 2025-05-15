import os
import numpy as np
import rasterio
import mapclassify
import pandas as pd

# ===== 0. Set working directory and parameters =====
folder = r'D:\cal_LST\area_cal'  # Folder containing LST images
output_csv = os.path.join(folder, 'LST_classification_statistics.csv')
k = 5  # Number of classification levels

# ===== 1. Initialize result list =====
all_results = []

# ===== 2. Loop through all tif images in the folder =====
for file in os.listdir(folder):
    if file.endswith('.tif'):
        filepath = os.path.join(folder, file)
        print(f'Processing: {file}')

        with rasterio.open(filepath) as src:
            lst = src.read(1)
            profile = src.profile
            transform = src.transform
            pixel_size = transform[0]
            area_per_pixel = pixel_size ** 2 / 1e6  # Pixel area in square kilometers

        # Remove invalid (nodata) values
        lst_valid = lst[lst != profile['nodata']]
        if lst_valid.size < k:
            print(f'Not enough valid data, skipping: {file}')
            continue

        # Classify using Natural Breaks method
        classifier = mapclassify.NaturalBreaks(lst_valid, k=k)
        bins = classifier.bins
        labels = np.full(lst.shape, -1, dtype=int)
        labels[lst != profile['nodata']] = classifier.yb

        # Construct result dictionary
        result = {"Image Name": file}
        for i in range(k):
            result[f"Class {i} Threshold (°C)"] = bins[i]
        for i in range(k):
            count = np.sum(labels == i)
            area = count * area_per_pixel
            result[f"Class {i} Area (km²)"] = round(area, 4)

        all_results.append(result)

# ===== 3. Save results as a CSV file =====
df = pd.DataFrame(all_results)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"All images processed. Statistics saved to: {output_csv}")
