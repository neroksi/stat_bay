
# Predicting pollution in China

*Authors: Nicolas Toussaint, Alexis Saïr, Robin Fuchs, Ambroise Coulomb, Enzo Terreau, Antoine Hoorelbeke*

## Introduction
This repository contains the implementation of the HMM algorithm describes in the [paper](https://econpapers.repec.org/article/eeeeconom/v_3a75_3ay_3a1996_3ai_3a1_3ap_3a79-97.htm) of Siddhartha Chib named "Calculating posterior distributions and modal estimates in Markov mixture models". You can find it [here](https://econpapers.repec.org/article/eeeeconom/v_3a75_3ay_3a1996_3ai_3a1_3ap_3a79-97.htm).

This repository is designed like a library to simulate and run HMM on different markov chain. You can define easily your prior and posterior. Then just run the HMM to get your results. Next, you can plot the ACF to see, how many steps from the beginning you need to remove.

## Examples
You can see two examples of the library working in the [examples folder](https://github.com/neroksi/stat_bay/tree/master/examples).

![A poisson mixture example](https://github.com/neroksi/stat_bay/raw/master/docs/poisson_mixture.png)

There are two notebooks :

 1. Poisson mixture : [pHMM.ipynb](https://github.com/neroksi/stat_bay/blob/master/examples/pHMM.ipynb)
 2. Gaussian mixture : [gHMM.ipynb](https://github.com/neroksi/stat_bay/blob/master/examples/gHMM.ipynb)

## Library structure
All the files running the library are in the [models folder](https://github.com/neroksi/stat_bay/tree/master/models).

 1. [posteriors.py](https://github.com/neroksi/stat_bay/blob/master/models/posteriors.py) : contains the implementation of different posteriors for the HMM. Posteriors are used to defined your model for the HMM
 2. [HMM.py](https://github.com/neroksi/stat_bay/blob/master/models/HMM.py) : contains the implementation of the HMM
 3. [simulation.py](https://github.com/neroksi/stat_bay/blob/master/models/simulation.py) : some function to simulate the markov chain from a posterior
 4. [utils.py](https://github.com/neroksi/stat_bay/blob/master/models/utils.py) : some utilities to plot the ACF.

