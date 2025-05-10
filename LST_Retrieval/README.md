# LST Retrieval - Landsat Single Channel Method 🌡️

This module contains Python code and notes for calculating Land Surface Temperature (LST) from Landsat 8/9 images using the single-channel algorithm.

## 📊 Data Source

All Landsat 8/9 images used in this project were downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/).

- Scene ID: LC08_L1TP_120038_20150902_20200908_02_T1
- Study Area: Nanjing, China
- Acquisition years: 2015, 2017, 2019, 2021, 2023

## 📌 Features
- Brightness temperature calculation from band 10
- Emissivity estimation using NDVI-based method
- Final LST conversion and unit adjustment
- Optional visualization with Matplotlib

## 🧰 Required Libraries
- rasterio
- numpy
- geopandas
- matplotlib

## 📁 Files
- `lst_single_channel.py`: main script for LST calculation
- `example_LST_2015.tif`: sample output
- `lst_workflow_notes.md`: optional explanation of steps

## 📖 Status
✅ Working version based on Nanjing LST project (2015–2023)

---

🧠 **Created by Guo Hang (郭航) — 2025**


This project is my first attempt to use GitHub, and also a step forward in improving my English.  
Since I didn't know how to use it, I asked ChatGPT for help — so this README is our joint effort!

Today marks a special moment: my first step in learning Github, Python, and English.  
Honestly, it's hard to express how I feel in English... maybe even in Chinese.  
But I hope I can keep going. One day, I might become truly great — who knows?!
