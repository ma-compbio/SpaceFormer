# Steamboat

[![Documentation Status](https://readthedocs.org/projects/spaceformer/badge/?version=latest)](https://spaceformer.readthedocs.io/en/latest/?badge=latest)

Steamboat is a computational model to decipher short- and long-distance cellular interactions based on the multi-head attention mechanism. 

![fig-github](https://github.com/user-attachments/assets/49284ea9-102c-4c13-bb77-b46bb7409c7e)

## Standard workflow
```python
import steamboat as sf # "sf" = "Steamboat Factorization"
import steamboat.tools
```

First, make a list (`adatas`) of one or more `AnnData` objects, and preprocess them.
```python
adatas = sf.prep_adatas(adatas, log_norm=True)
dataset = sf.make_dataset(adatas)
```

Create a `Steamboat` model and fit it to the data.
```python
model = sf.Steamboat(adata.var_names.tolist(), d_ego=0, d_local=5, d_global=0)
model = model.to("cuda") # if you GPU acceleration is supported.
model.fit(dataset, masking_rate=0.25)
```

After training you can check the trained metagenes.
```python
sf.tools.plot_transforms(model, figsize=(4, 4), vmin=0.0, vmax=.5, top=0)
```
This can be very slow for a lot of metagenes, so you can use another function to just plot one of them.
```python
plot_transform(model, "local", 0, top=18)
```

For clustering and segmentation, run the following lines. Change the resolution to your liking.
```python
sf.tools.neighbors(adata)
sf.tools.leiden(adata, resolution=0.1)
sf.tools.segment(adata, resolution=0.5)
```

A [tiny simulation example](https://spaceformer.readthedocs.io/en/latest/tutorial_nbs/tiny_simulation_example.html) is available in our documentation.

## Documentation
For the full API and real data examples, please visit our [documentation](https://spaceformer.readthedocs.io/en/latest/).

