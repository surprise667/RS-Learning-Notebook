# 🌡️ Land Surface Temperature (LST) Classification using Natural Breaks

This script performs land surface temperature (LST) classification using the Natural Breaks (Jenks) method. 
It is designed for processing LST images (e.g., derived from Landsat) and generating a categorized image for visualization or further spatial analysis.

---

## 📌 Features

- Reads a LST GeoTIFF image (in degrees Celsius)
- Removes invalid (nodata) pixels
- Applies Natural Breaks classification (default: 5 classes)
- Outputs a classified GeoTIFF image with integer class labels
- Prints classification thresholds and output path

---

## 🛠️ Dependencies

```bash
pip install rasterio numpy mapclassify matplotlib
```

---

## 🚀 How to Use

1. Modify the `lst_path` variable in the script to your LST image file.
2. Run the script with Python.
3. The classified result will be saved to a new file (e.g., `LST_classified.tif`).

---

## 📂 Output

- A GeoTIFF file with pixel values ranging from 0 to 4, representing the class each pixel belongs to.
- Example:
  ```
  🌡️ LST classification thresholds (°C): [27.5, 30.2, 32.9, 35.6, 38.3]
  ✅ Classified LST image saved to: D:/.../LST_classified.tif
  ```

---

## 📎 Notes

- Make sure the input LST image has proper `nodata` metadata.
- You can change the number of classes by modifying `k=5` in the script.
- This script is part of a broader workflow for analyzing urban heat islands using remote sensing.

