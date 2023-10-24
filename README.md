# Code for "Beyond Bayesian Model Averaging over Paths in Probabilistic Programs with Stochastic Support"

This is the code to reproduce the experiments of the paper [Beyond Bayesian
Model Averaging over Paths in Probabilistic Programs with Stochastic Support](https://arxiv.org/abs/2310.14888).

The code uses Poetry to manage dependencies. To create a virtual environment go to [https://python-poetry.org/docs/](https://python-poetry.org/docs/) and follow the installation instructions.

Then do `poetry install` to install all the necessary project dependencies.

Scripts to reproduce the plots are in `plots`.

Code for the RJMCMC baseline implemented in Gen is provided in `rjmcmc_gen`. Note we ran the code with Julia version 1.3.1. Different versions of Julia might not be compatible.

## When is Stacking Helpful?

Distinct SLPs:
```
cd scripts
poetry run python stacking_distinct_models.py --num-replications=1000
```

SLPs with overlap:
```
cd scripts
poetry run python stacking_overlapping_model.py \
    --num-replications=1000 \
    --coeff3=0.3 \
    --coeff4=0.1 \
    --cov-mean=0.0
```


Dominating SLPs:
```
cd scripts
poetry run python stacking_dominating_model.py --num-replications=1000
```

## Subset Regression

```
poetry run python variable_selection.py \
    --num-replications=10
```

## Function Induction

```
poetry run python function_induction.py \
    -m seed=0,1,2,3,4,5,6,7,8,9 \
    name=fun_ind \
    dcc_hmc.num_parallel=32 \
    dcc_hmc.num_mcmc=500 \
    dcc_hmc.num_mcmc_warmup=200 \
    dcc_hmc.max_slps=128 \
    dcc_hmc.num_chains=4 \
    num_train=200 \
    num_val=200 
```

## Variable Selection

California dataset:
```
poetry run python variable_selection.py \
    -m seed=0,1,2,3,4,5,6,7,8,9 \
    name=var_select_california \
    dcc_hmc.num_parallel=16 \
    dcc_hmc.num_mcmc=1000 \
    dcc_hmc.num_mcmc_warmup=400 \
    dataset.num_train=8260 \
    dataset.num_val=2065 \
    dataset.num_test=10325
```

Stroke dataset:
```
poetry run python variable_selection.py \
    -m seed=0,1,2,3,4,5,6,7,8,9 \
    name=var_select_stroke \
    dcc_hmc.num_parallel=12 \
    dcc_hmc.num_mcmc=1000 \
    dcc_hmc.num_mcmc_warmup=400 \
    dataset=stroke \
    model=log_reg
```

Diabetes dataset:
```
poetry run python variable_selection.py \
    -m seed=0,1,2,3,4,5,6,7,8,9 \
    name=var_select_diabetes \
    dcc_hmc.num_parallel=16 \
    dcc_hmc.num_mcmc=1000 \
    dcc_hmc.num_mcmc_warmup=400 \
    dataset=diabetes \
    model=log_reg
```

## Modelling Radon Contamination in US Counties

```
poetry run python radon.py \
    -m seed=0,1,2,3,4,5,6,7,8,9 \
    name=radon \
    dcc_hmc.num_parallel=12 \
    dcc_hmc.num_mcmc=2000 \
    dcc_hmc.num_mcmc_warmup=2000
```
