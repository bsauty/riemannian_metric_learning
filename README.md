# Riemannian metric learning

This repo proposes a framework to evaluate longitudinal evolution of scalar data with a mixed effect model. 
The goal is to both recover the average trajectory of the features and the individual variations to this 
mean trajectory for each individual in order to best reconstruct the trajectories. The goal is threesome : get a better
understanding of the **average evolution** of the features and allow to fit every patients to a personalize trajectory in order 
to perform **imputation** of missing data and **prediction** for future timepoints.

This method was presented and published at the International Symposium for Biomedical Imaging 2022 (link to come).

The data is supposed to lie on a Riemannian manifold on which trajectories are small variations around a reference geodesic. This repo allows to choose the metric for the fit between logistic, linear, exponential, or parametric metric. The last one is the main contribution of this paper and is iteratively learned along the MCMC procedure. 

## Quick overview of the file tree

This implementation is based off of the architecture of the [deformetrica](https://www.deformetrica.org/) software, as it provides geometric tools such as the computation of efficient parallel transport or the numerical computation of geodesics on a Riemannian manifold.

- `api` : contains the deformetrica api that handles loggers and hardware optimization.

- `core` : 
  - `estimator_tools` :  sampler used to get realisations of the individual parameters for the Metropolis
    Hastings Gibbs procedure.
  - `estimators` : the general classes of estimators used `mcmc_saem`, `gradient_ascent` and
  `scipy_minimize`. These are used to recover the fixed effects and random effects.
  - `model_tools` : Riemannian tools. It defines the `exponential` and `geodesic` objects and an object called
  `spatiotemporal_reference_frame` that contains the reference geodesic, the two exponentials (forward and backward wrt the reference time) and all 
    the numerical solvers for Hamiltonian equations (for both parallel transport and exponentiation).
  - `models` : the main model we use is `longitudinal_metric_learning`. It wraps the dataset, all the individual parameters that are sampled in the MCMC and 
    all the fixed effects that are optimized in the EM phase.
  - `observations` : all the tools to load the dataset.

More specifically in `parametric/` there are 3 relevant files :
- `initialize_parametric_model.py` : first rough estimation of the parameters through a gradient descent
  in order to 'help' the MCMC algorithm. This is full of heuristics and data-dependant smart initialization. It is not a necessary step, although much recommended.
- `fit_parametric_model.py`: once the initialization is done, we launch the MCMC-SAEM procedure to fit the model.
- `personalize_parametric_model.py` : after the fit we may want to estimate the individual parameters for patients that were not in 
the training dataset, to evaluate the model's ability to generalize. This is a simple gradient descent on the individual parameters
  while keeping the fixed effects as they were learned during the fit.

These 3 methods require the arguments `dataset.xml`/`optimization_parameters.xml`/`model.xml` where
- `dataset.xml` : contains the paths to the dataset. Namely a dataset consists of 3 csv files : the times of visits, the id of patients for each visit and the corresponding biomarkers values. 
- `optimization_parameters.xml` : contains the hyperparameters for the MCMC and gradient ascent procedures.
- `model.xml` : contains the hyperparameters for the model (mostly the type of metric used and, if parametric, the width of the kernel. 

## TODO

- [ ] Reorganise the way arguments are passed to the model because the `dataset.xml`/`optimization_parameters.xml`/`model.xml`is
not very straightforward.
- [ ] Pass datasets as one csv and parse it.


## References

