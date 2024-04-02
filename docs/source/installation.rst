Installation
=======================================

A conda virtual environment is highly recommended when training SpaceFormer. 
After you are at the root path of this directory, 
use the following commands to set up a proper virtual environment::

   conda create -n SpaceFormer python=3.8
   conda activate SpaceFormer

To use CUDA for model fitting, follow the `instructions <https://pytorch.org/get-started/locally/>`_ to install PyTorch.

You can then use pip to install SpaceFormer and all the dependencies::

   git clone https://github.com/ma-compbio/SpaceFormer
   cd SpaceFormer
   git install .

