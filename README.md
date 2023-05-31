# Experiments of the paper Risk of Recourse in Binary Classification

This repository contains all the relevant code to run the experiments that are conducted in the paper 
'The Risks of Recourse in Binary Classification'.

## Installing the requirements

The requirements can either be installed using the `requirements.txt` file or the `cor.yml` file.

The first option:
```
pip install -r requirements.txt
```

The second option:
```
conda env create -f cor.yml
```

The `cogs` package needs to be installed separately:
```
git clone git@github.com:marcovirgolin/CoGS.git
cd CoGS
pip install .
```

## Running the experiments
The experiments are divided in two parts. The synthetic data experiments and the real data experiments. 

### Synthetic Data
Each experiment can be started using the `synthetic_experiments.py` file. Use the 
`--data_set`, `--classifier`, `--recourse` flags to indicate which experiment you want to run. An example would be
```
python -u synthetic_experiments.py  \
    --data_set moons \
    --classifier lr \
    --recourse brute_force
```

#### Linear Gaussians
There is an additional file, `linear_gaussians.py`, which runs the experiment for the Gaussian example (3.1 in the main paper). This function can be run without any parameters 


#### Conditional Probability plots
There is one final additional file, `synthetic_conditional_plot.py`, which calculates and plots the conditional
true probabilities of the `moons`, `circles`, and `gaussians` synthetic data sets.

### Real Data
Each experiment can be started using the `real_experiments.py` file. Use the 
`--data_set`, `--classifier`, `--recourse` flags to indicate which experiment you want to run. An example would be
```
python -u real_experiments.py  \
    --data_set adult \
    --classifier gbc \
    --recourse wachter 
```


## Scripts

The main directory contains two scripts, `run_synth_experiments.sh` and `run_real_experiments.sh` that runs
all experiments in parralel. WARNING! The real data experiment can take a very long time, even when running in parallel. 
