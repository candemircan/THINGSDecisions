# Decision-Making with Naturalistic Options
This repository contains the code for [Demircan et al., 2022](https://escholarship.org/content/qt1td8q3wn/qt1td8q3wn.pdf) where we studied how people make decisions in a two-alternative forced choice paradigm with high-dimensional, naturalistic stimuli.
## Reproducing the Results
You need to have `conda` & `r` installed on your machine. When you run `source main.sh` virtual environments containing the packages that were used in our analyses are created, and all the code to get from raw data to the presented results is run. 

Notice that a few analysis functions (to run the computational models and do the leave-one-out cross validation analyses) have an optional `remake` parameter, for which the default is set to `False`. Under this condition, if the necessary results file exists, it is load into the memory and the analysis is not run from scratch. This was done in order to save time, however if you want to run the analyses from scratch, set these parameters to `True`.
