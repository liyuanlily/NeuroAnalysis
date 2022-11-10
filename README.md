# Code for Replay Analysis
This is a realization of several analysis methods for hippocampal replay, including Bayesian Decoder, Gerneral Linear Model, and reactivation analysis, cluster identification, assembly analysis methods. These methods are for both 2P imaging and Neuropixels data. And some of the methods originated from recently published papers.

The programming language is python.

## Installation
You need to install the following python packages:
* numpy
* scipy
* pytorch
* sklearn
* matplotlib

You can download the .py file and import functions in it to your python code.

## Usage
For a more detailed illustration, please see Wiki (to be complete). 

### Preprocessing Data
After deconvolving 2P imaging or Neuropixels data, you can use functions in PreprocessData.py to adjust data into desired format for analysis afterwards.

### Reactivation Analysis
After preprocessing the neural activity data, you want to detect High Synchrony Events (HSEs), find reactivated cells in each HSE, classify reactivated cells into assemblies and analysis the properties of the assemblies. 

Needed functions can be found in ReactivationAnalysis.py, ClusterIdentification.py and AssemblyAnalysis.py.

In GoodHSEs.py, we provide a way to fine-tuning the parameters of the HSE detction function. Firstly, we need manually labelled samples of a HSE to be good or not. After training a neural network, we can then use different parameters for HSE detection, and see which combination generates as many as good HSEs as it can without losing a big deal of it.

### Replay Analysis
You can train a Bayesian Decoder on the running periods, predict the mouse's running trajectory in each HSE and find replay events accordingly. 

Needed functions can be found in BayesianDecoder.py.

### GLM Encoder
This is a realization of the GLM encoder for neural coding in Minderer, 2019.

GLMEncoder.py is a cpu version based on numpy and scipy, while GLMTorch.py is a gpu version based on pytorch. The encoder runs much faster in pytorch.
