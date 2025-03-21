# Project Directory Structure

This document outlines the folder layout of the **SAR Eddy Detection Demo** project. It explains the purpose of each directory and helps you navigate the repository more easily.

## Directory Tree

```
config/
├── default.yaml        # Global default settings
└── inference.yaml      # Settings specific to running inference

data/
├── land_mask/
│   ├── ne_10m_land.shp # Natural Earth land boundary shapefile
│   └── ne_10m_land.shx # Spatial index for land shapefile
└── README.md           # Instructions for data handling

model_checkpoints/
└── checkpoint.tar      # Pretrained model checkpoint

src/
├── __init__.py         # Initializes the src directory as a package
├── dataset.py          # Data handling and preprocessing routines
├── inference.py        # Model inference pipeline logic
├── main.py             # Main entry point for running the demo
├── model.py            # Model loading and architecture (supports r50_1x_sk0)
├── visualize_eddy_bbox.py  # Visualization tools for detected eddy bounding boxes
├── models/
│   ├── __init__.py     # Initializes the models package
│   └── simclr_resnet.py # SimCLR ResNet implementation
└── utils/
    ├── __init__.py     # Initializes the utilities package
    ├── bbox.py         # Bounding box manipulation functions
    └── config.py       # Configuration parsing functions

demos/
├── preview_demo.ipynb  # Notebook for previewing image outputs
└── run_demo.sh         # Shell script for running the demo pipeline

output/
└── (All output results, such as visualizations and logs, will be saved here)
```

## Folder Descriptions

- **config/**  
  Contains YAML configuration files:
  - `default.yaml`: Contains global default settings.
  - `inference.yaml`: Contains settings specific to running inference.

- **data/**  
  Stores all input data files:
  - The `land_mask` folder includes the Natural Earth land boundary shapefile and its spatial index.
  - The README in this folder provides additional instructions for managing data.

- **model_checkpoints/**  
  Contains the pretrained model checkpoint (`checkpoint.tar`) used during inference.

- **src/**  
  Holds all source code for the project:
  - `main.py` is the primary entry point.
  - Other scripts (`dataset.py`, `inference.py`, `model.py`, and `visualize_eddy_bbox.py`) implement core functionalities.
  - The `models/` subfolder contains specific model implementations.
  - The `utils/` subfolder contains helper functions for configuration parsing and bounding box operations.

<!-- - **demos/**  
  Provides demonstration files:
  - A Jupyter Notebook (`preview_demo.ipynb`) to preview outputs.
  - A shell script (`run_demo.sh`) to run the full pipeline. -->

- **output/**  
  Destination folder for all generated results (e.g., detection outputs, visualizations).

## Usage

For detailed instructions on setting up your environment and running the demo, please refer to the [Installation Guide](INSTALLATION.md). This document focuses on the repository layout to help you understand where each component is located.
