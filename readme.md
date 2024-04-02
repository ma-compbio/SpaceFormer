# SpaceFormer
SpaceFormer is a Transformer based computational tool for spatial omics analysis. Given a set of adata files, this document demonstrates how one can train a SpaceFormer in a self-supervised manner.

## Dependencies
A conda virtual environment is highly recommended when training SpaceFormer. After you are at the root path of this directory, use the following commands to set up a proper virtual environment:
```bash
conda create -n SpaceFormer python=3.8
conda activate SpaceFormer
pip install -r requirements.txt
```

## Tutorials (Jupyter Notebooks)
- [Build a dataset from `AnnData` and train a SpaceFormer model](https://github.com/ma-compbio/SpaceFormer/blob/main/notebooks/codex_training.ipynb)
- [Analyze trained model at sample level](https://github.com/ma-compbio/SpaceFormer/blob/main/notebooks/codex_analysis_one_sample.ipynb)
- [Analyze trained model across samples](https://github.com/ma-compbio/SpaceFormer/blob/main/notebooks/codex_analysis_across_samples.ipynb)


## Documentation
[ReadTheDocs](https://spaceformer.readthedocs.io/en/latest/)
