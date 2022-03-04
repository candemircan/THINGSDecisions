#!/bin/bash

conda init

if ! conda env list | grep ".*things_decisions.*"
then
	conda env create -f things_decisions.yml
fi

conda activate things_decisions

python src/Py/main_part1.py

Rscript -e "renv::restore()"
Rscript src/R/main.R

python src/Py/main_part2.py
