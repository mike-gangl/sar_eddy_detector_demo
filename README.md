# SAR Eddy Detection Demo

This repository contains a demo for detecting ocean eddies from Synthetic Aperture Radar (SAR) imagery. The demo uses a pretrained model (`r50_1x_sk0`) to process SAR GeoTIFF files, identify eddy features, and generate output visualizations.


## Directory Structure

```
config/
├── default.yaml (Global default settings)
└── inference.yaml (Settings specific to running inference)
data/
├── land_mask
│   ├── ne_10m_land.shp (Natural Earth land boundary shapefile)
│   └── ne_10m_land.shx (Spatial index for land shapefile)
└── README.md (Instructions for data)
model_checkpoints/
└── checkpoint.tar (Pretrained model checkpoint)
src/
├── __init__.py (Initializes the src directory as a package)
├── dataset.py (Data handling and preprocessing)
├── inference.py (Model inference pipeline logic)
├── main.py (Main entry point)
├── model.py (Currently supports only r50_1x_sk0 model)
├── visualize_eddy_bbox.py (Tools for visualizing detected eddy bounding boxes)
├── models/
│   ├── __init__.py (Package initialization for models)
│   └── simclr_resnet.py (SimCLR ResNet implementation)
└── utils/
    ├── __init__.py (Package initialization for utilities)
    ├── bbox.py (Bounding box manipulation functions)
    └── config.py (Configuration parsing functions)
demos/
├── preview_demo.ipynb (Placeholder notebook for image previews)
└── run_demo.sh (Shell script for running the pipeline)
output/
└── (Directory for output results)
```

## Getting Started
### Installation

You can set up the environment using either conda or venv/pip.

#### Option 1: Conda (Recommended)

We recommend using conda to manage your environment. Conda allows for easy installation of scientific packages, especially those with binary dependencies.

**1. Install Conda:**

If you don't have conda installed, follow the instructions for your operating system from [Miniforge](https://conda-forge.org/download/).

#### Installing Miniforge

Miniforge provides a minimal conda installation tailored for the `conda-forge` ecosystem.  
Choose the correct installer for your operating system and architecture from the table below:

| Operating System       | Architecture | Download Command                                                                                                          | Installation Command                                    |
|------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| **Linux**             | x86_64      | `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh`           | `bash Miniforge3-Linux-x86_64.sh`           |
| **Linux**             | aarch64     | `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh`          | `bash Miniforge3-Linux-aarch64.sh`          |
| **macOS** (Intel)     | x86_64      | `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh`          | `bash Miniforge3-MacOSX-x86_64.sh`          |
| **macOS** (Apple Sil.)| arm64       | `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`           | `bash Miniforge3-MacOSX-arm64.sh`           |
| **Windows**           | x86_64      | `curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"` | Double-click the `.exe` and follow on-screen prompts     |

> **Note for Windows users:**  
> You can run the `curl` command from a PowerShell terminal. After the download finishes, run (double-click) the `.exe` installer and follow the on-screen instructions to complete installation.


**2. Create a Conda Environment:**

Open your terminal or Anaconda Prompt and use the following commands to create a conda environment named `sar_eddy_env`. These instructions are the same for Mac, Windows, and Linux.

```bash
conda create -n sar_eddy_env python=3.10
conda activate sar_eddy_env
```

**3. Install Required Packages:**
```
conda install numpy pandas matplotlib rasterio geopandas shapely tqdm pyyaml -c conda-forge

conda install pytorch torchvision -c conda-forge
```

**Platform-specific notes for Conda:**

- **Windows Users:** If you encounter issues with `rassterio` installation via conda, try installing it with pip after activating your conda environment: `pip install rasterio`
- **Mac Users (Apple Silicon):** For improved performance on Apple Silicon (M1/M2/M3) Macs, install the accelerated `libblas` library: `conda install "libblas-*=*accelerate" -c conda-forge
- **GPU Support (PyTorch):** The above command installs the CPU version of PyTorch. For GPU support, you need to install the CUDA-enabled version. Follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to find the correct command for your CUDA version.

#### Option 2: venv and pip

If you prefer not to use conda, you can use Python's built-in `venv` for environment management and `pip` for package installation.

**1. Create a Virtual Environment:**

Open your terminal and navigate to the root directory of the repository. Create a virtual environment:

```bash
python -m venv venv
```

**2. Activate the Virtual Environment:**

Activate the virtual environment for your operating system:

*   **Linux/Mac:**
    ```bash
    source venv/bin/activate
    ```
*   **Windows:**
    ```bash
    venv\Scripts\activate
    ```

**3. Upgrade pip and Install Required Packages using pip:**

It's recommended to upgrade pip before installing dependencies. Then, install the required packages from the `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**(Optional) Using `setup.py`:**

Users can install the project and its dependencies using pip with the following command from the repository root directory:

```bash
pip install -e .
```

This command installs the project in editable mode ( `-e .`), meaning changes to the `src` directory will be immediately reflected without reinstalling.


### Configuration

Edit the configuration files in the config/ folder if needed:
- **inference.yaml**: Contains settings specific to inference (e.g., batch size, model checkpoint path).

### Data

1.  **Land Mask:** Download the land shapefile (ne_10m_land.shp) and place it in the data/land_mask/ directory. You can obtain this from [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/).
2.  **SAR Tile:** Obtain a sample SAR GeoTIFF tile and rename it to $exampleTileFileName. Place this file in the data/ directory. For demonstration purposes, you can use a small tile.

### Running the Demo

You can run the demo pipeline using:

  ```bash
  python src/main.py --config config/inference.yaml
  ```

This script calls the main entry point (in src/main.py) which:
1. Loads the configuration.
2. Sets up the dataset (using src/dataset.py).
3. Loads the pretrained ResNet-50 model (via src/model.py).
4. Runs inference to detect ocean eddies.
5. Saves detection outputs and (optionally) generates preview images.

<!-- ### Interactive Demo

For a more interactive demonstration, open the placeholder Jupyter Notebook:

  ```bash
  jupyter notebook demos/preview_demo.ipynb
  ```
**Note:** demos/preview_demo.ipynb is a placeholder file. You can replace it with a Jupyter Notebook to show preview image generation if you implement that feature. -->

## Simplified Model Loading

For this demo, we only support the ResNet-50 (`r50_1x_sk0`) model architecture. The model loading process is encapsulated in [src/model.py](src/model.py) and is designed to be easily extended to support additional architectures in the future.

## Checkpoint and Example Data

- **Checkpoint:** The `model_checkpoints/` directory should contain the model checkpoint file named $modelFileName.
- **Example SAR Tile:** The data/ directory should contain the example SAR GeoTIFF tile named $exampleTileFileName.

<!-- ## Future Work

- **Support Multiple Models:** Extend src/model.py to handle additional architectures and checkpoints.
- **Enhanced Logging:** Integrate a logging framework for better runtime diagnostics.
- **Deployment Pipeline:** Build a full-fledged operational system with web interfaces or cloud integration.
- **More Robust Data Handling:** Add data validation and preprocessing improvements.
- **Implement Preview Image Generation:** Create a Jupyter Notebook (demos/preview_demo.ipynb) to demonstrate visualization of detection results.

## License

[MIT License](LICENSE) -->
