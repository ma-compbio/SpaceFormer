import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .model import SpaceFormer
from typing import Literal
from torch import nn
import scanpy as sc

def plot_transforms(model: SpaceFormer):

    if model.d_global > 0:
        pass
    

def score_cells(model: SpaceFormer):
    pass

