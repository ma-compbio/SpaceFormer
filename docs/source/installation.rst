Installation
=======================================

A conda virtual environment is highly recommended when training Steamboat. 
After you are at the root path of this directory, 
use the following commands to set up a proper virtual environment::

   conda create -n steamboat python=3.11
   conda activate steamboat

To use CUDA for model fitting, follow the `instructions <https://pytorch.org/get-started/locally/>`_ to install PyTorch.

You can then use pip to install Steamboat and all the dependencies::

   git clone https://github.com/ma-compbio/Steamboat.git
   cd Steamboat
   git install .

