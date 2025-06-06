from condaforge/miniforge3:25.3.0-3
#from continuumio/miniconda3

run conda create -n sar_eddy_env python=3.10
run conda install -n sar_eddy_env pandas rasterio libgdal "geopandas>1.0.0" shapely tqdm pyyaml -c conda-forge -y --quiet
run conda install -n sar_eddy_env pytorch torchvision -c conda-forge -y --quiet


copy . .

CMD ["conda" , "run", "-n", "sar_eddy_env", "python", "src/main.py", "--config", "config/inference.cpu.yaml"]
