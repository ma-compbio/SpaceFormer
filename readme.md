# SapceFormer
SpaceFormer is a Transformer based computational tool for spatial omics analysis. Given a set of adata files, this document demonstrates how one can train a SpaceFormer in a self-supervised manner.

## Dependencies
A conda virtual environment is highly recommended when training SpaceFormer. After you are at the root path of this directory, use the following commands to set up a proper virtual environment:
```bash
conda create -n SpaceFormer python=3.8
conda activate SpaceFormer
pip install -r requirements.txt
```
