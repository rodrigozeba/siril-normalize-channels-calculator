# Normalize Channels Calculator for Siril

A PyQt6 Python script for Siril that estimates narrowband SHO channel weights using median and percentile-based signal estimation.
Recommended: use starless mono channels when possible.
Bright stars can bias percentile-based signal estimates, especially P95/P98.

## Features

- Select Ha, SII and OIII mono FITS files
- Calculate median, percentile and estimated signal
- Generate starting SHO Pixel Math formulas
- Edit formulas manually
- Apply formulas and create an RGB FITS composition
- Save and open the result in Siril
- Simple autostretched preview

## Requirements

- Siril 1.4+
- Python scripting enabled in Siril
- PyQt6 support in Siril's Python environment
- NumPy

## Basic workflow

1. Open the script from Siril's Scripts menu.
2. Select Ha, SII and OIII mono FITS files.
3. Choose a percentile, usually 90, 95 or 98.
4. Click Calculate.
5. Adjust formulas if needed. (Suported operations: med() median() mean() min() max() abs() sqrt() log() log10() asinh() clip())
6. Click Apply.
7. The script creates `normalized_SHO_result.fit`.
8. Use the Zoom tools to zoom in and out, fit to preview, scale 1:1 and pan

## Notes

The preview uses a simple linked autostretch for display only. The saved FITS is the composed RGB result.