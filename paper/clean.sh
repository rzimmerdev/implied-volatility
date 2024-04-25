#!/bin/bash

# Define the directory where your TEX files are located
TEX_DIR="./"

# Clean auxiliary TEX build files
echo "Cleaning TEX build files in $TEX_DIR..."
find "$TEX_DIR" -type f \( -name "*.aux" -o -name "*.log" -o -name "*.synctex.gz" -o -name "*.toc"  -o -name "*.blg"  -o -name "*.out"  -o -name "*.bbl" \) -delete

echo "TEX build files cleaned successfully!"