#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import os.path
import time

import torch
import sys

from copy import deepcopy
import torch.nn as nn
import torch.optim as optim

sys.path.append('/home/benoit.sautydechalon/deformetrica')
import deformetrica as dfca

from deformetrica.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from deformetrica.core.estimators.gradient_ascent import GradientAscent
from deformetrica.core.estimators.mcmc_saem import McmcSaem
# Estimators
from deformetrica.core.estimators.scipy_optimize import ScipyOptimize
from deformetrica.core.model_tools.manifolds.exponential_factory import ExponentialFactory
from deformetrica.core.model_tools.manifolds.generic_spatiotemporal_reference_frame import GenericSpatiotemporalReferenceFrame
from deformetrica.core.model_tools.neural_networks.networks import Dataset, CVAE, CAE, LAE
from deformetrica.in_out.array_readers_and_writers import *
from deformetrica.core import default
from deformetrica.in_out.dataset_functions import create_image_dataset_from_torch, create_scalar_dataset
from deformetrica.support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from deformetrica.support.utilities.general_settings import Settings
from deformetrica.core.models import LongitudinalAutoEncoder

from deformetrica.support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_spatiotemporal_reference_frame(model, logger, observation_type='image'):
    """
    Initialize everything which is relative to the geodesic its parameters.
    """
    exponential_factory = ExponentialFactory()
    exponential_factory.set_manifold_type('euclidean')
    logger.info('Initialized the Euclidian metric for latent space')

    model.spatiotemporal_reference_frame = GenericSpatiotemporalReferenceFrame(exponential_factory)
    model.spatiotemporal_reference_frame.set_concentration_of_time_points(default.concentration_of_time_points)
    model.spatiotemporal_reference_frame.set_number_of_time_points(default.number_of_time_points)
    model.no_parallel_transport = False
    model.spatiotemporal_reference_frame.no_parallel_transport = False
    model.number_of_sources = Settings().number_of_sources

def initialize_CAE(logger, model, path_CAE=None):

    if (path_CAE is not None) and (os.path.isfile(path_CAE)):
        checkpoint =  torch.load(path_CAE, map_location='cpu')
        autoencoder = CAE()
        autoencoder.load_state_dict(checkpoint)
        logger.info(f">> Loaded CAE network from {path_CAE}")
    else:
        path_CAE = 'CAE'
        logger.info(">> Training the CAE network")
        epochs = 500
        batch_size = 6
        lr = 1e-4

        autoencoder = CAE()
        logger.info(f"Learning rate is {lr}")
        logger.info(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

        # Load data
        train_loader = torch.utils.data.DataLoader(model.train_images, batch_size=batch_size,
                                                   shuffle=True, num_workers=10, drop_last=True)
        criterion = nn.MSELoss()
        optimizer_fn = optim.Adam
        optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
        autoencoder.train(train_loader, test=model.test_images, criterion=criterion,
                          optimizer=optimizer, num_epochs=epochs)
        logger.info(f"Saving the model at {path_CAE}")
        torch.save(autoencoder.state_dict(), path_CAE)

    return autoencoder

def initialize_LAE(logger, model, path_LAE=None):

    if (path_LAE is not None) and (os.path.isfile(path_LAE)):
        checkpoint =  torch.load(path_LAE, map_location='cpu')
        autoencoder = LAE()
        autoencoder.load_state_dict(checkpoint)
        logger.info(f">> Loaded LAE network from {path_LAE}")
    else:
        path_LAE = 'LAE'
        logger.info(">> Training the LAE network")
        epochs = 20
        batch_size = 4
        lr = 0.000001

        autoencoder = LAE()
        logger.info(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

        train_loader = torch.utils.data.DataLoader(model.train_encoded, batch_size=batch_size,
                                                   shuffle=True, num_workers=4, drop_last=True)
        criterion = nn.MSELoss()
        optimizer_fn = optim.Adam
        optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
        autoencoder.train(train_loader, test=model.test_encoded, criterion=criterion,
                          optimizer=optimizer, num_epochs=epochs)
        torch.save(autoencoder.state_dict(), path_LAE)

    return autoencoder

def instantiate_longitudinal_auto_encoder_model(logger, path_data, path_CAE=None, path_LAE=None,
                                                dataset=None, xml_parameters=None, number_of_subjects=None):

    model = LongitudinalAutoEncoder()

    # Load the train/test data
    torch_data = torch.load(path_data)
    torch_data = Dataset(torch_data['data'].unsqueeze(1), torch_data['target'])
    train, test = torch_data[:len(torch_data) - 200], torch_data[len(torch_data) - 200:]
    train, test = Dataset(train[0], train[1]), Dataset(test[0], test[1])
    logger.info(f"Loaded {len(train.data)} train images and {len(test.data)} test images")
    model.train_images, model.test_images = train.data, test.data
    model.train_labels, model.test_labels = train.labels, test.labels
    # TODO : put this as an attribute of the model ?
    criterion = nn.MSELoss()

    # Initialize the CAE
    model.CAE = initialize_CAE(logger, model, path_CAE=path_CAE)

    # TODO : delete this after debugging LAE training
    if not os.path.isfile('encoded_dataset'+path_CAE):
        logger.info("Saving the encoded dataset for training purposes")
        _, encoded_data = model.CAE.evaluate(model.train_images.dataset.data, criterion)
        to_save = {'data': encoded_data, 'target':model.train_images.dataset.labels}
        torch.save(to_save, 'encoded_dataset' + path_CAE)
        logger.info(f"Saved the encoded to 'encoded_dataset' -> {encoded_data.shape}")
    else:
        logger.info("Encoded dataset is already saved at 'encoded_dataset'")

    # Then initialize the first latent representation
    _, model.train_encoded = model.CAE.evaluate(model.train_images, criterion)
    _, model.test_encoded = model.CAE.evaluate(model.test_images, criterion)

    # Initialize the LAE
    model.LAE = initialize_LAE(logger, model, path_LAE=path_LAE)
    model.observation_type = 'scalar'

    # Then initialize the actual latent variables that constitute our longitudinal dataset
    train_data = Dataset(model.train_encoded, model.train_labels)
    _, initial_latent_representation = model.LAE.evaluate(train_data, criterion)
    group, timepoints = [label[0] for label in model.train_labels], [label[1] for label in model.train_labels]
    dataset = create_scalar_dataset(group, initial_latent_representation.numpy(), timepoints)

    # Then initialize the longitudinal model for the latent space
    if dataset is not None:
        template = dataset.deformable_objects[0][0] # because we only care about its 'metadata'
        model.template = deepcopy(template)

    # Initialize the fixed effects, either from files or to arbitrary values
    if xml_parameters is not None:
        # Reference time
        model.set_reference_time(xml_parameters.t0)
        model.is_frozen['reference_time'] = xml_parameters.freeze_reference_time
        # Initial velocity
        initial_velocity_file = xml_parameters.v0
        model.set_v0(read_2D_array(initial_velocity_file))
        model.is_frozen['v0'] = xml_parameters.freeze_v0
        # Initial position
        initial_position_file = xml_parameters.p0
        model.set_p0(read_2D_array(initial_position_file))
        model.is_frozen['p0'] = xml_parameters.freeze_p0
        # Time shift variance
        model.set_onset_age_variance(xml_parameters.initial_time_shift_variance)
        model.is_frozen['onset_age_variance'] = xml_parameters.freeze_onset_age_variance
        # Log acceleration variance
        model.set_log_acceleration_variance(xml_parameters.initial_acceleration_variance)
        model.is_frozen["log_acceleration_variance"] = xml_parameters.freeze_acceleration_variance
        # Noise variance
        model.set_noise_variance(xml_parameters.initial_noise_variance)
        # Modulation matrix
        modulation_matrix = read_2D_array(xml_parameters.initial_modulation_matrix)
        if len(modulation_matrix.shape) == 1:
            modulation_matrix = modulation_matrix.reshape(Settings().dimension, 1)
        logger.info(f">> Reading {str(modulation_matrix.shape[1]) }-source initial modulation matrix from file: {xml_parameters.initial_modulation_matrix}")
        assert xml_parameters.number_of_sources == modulation_matrix.shape[1], "Please set correctly the number of sources"
        model.set_modulation_matrix(modulation_matrix)
        model.number_of_sources = modulation_matrix.shape[1]

    else:
        # All these initial paramaters are pretty arbitrary TODO: find a better way to initialize
        model.set_reference_time(70)
        model.set_v0(np.ones(Settings().dimension)/30)
        model.set_p0(np.zeros(Settings().dimension))
        model.set_onset_age_variance(5)
        model.set_log_acceleration_variance(0.1)
        model.number_of_sources = Settings().number_of_sources
        modulation_matrix = np.zeros((Settings().dimension, model.number_of_sources))
        model.set_modulation_matrix(modulation_matrix)
        model.initialize_modulation_matrix_variables()

    # Initializations of the individual random effects
    assert not (dataset is None and number_of_subjects is None), "Provide at least one info"

    if dataset is not None:
        number_of_subjects = dataset.number_of_subjects

    # Initialization the individual parameters
    if xml_parameters is not None:
        logger.info(f"Setting initial onset ages from {xml_parameters.initial_onset_ages} file")
        onset_ages = read_2D_array(xml_parameters.initial_onset_ages).reshape((len(dataset.times),))
        logger.info(f"Setting initial log accelerations from { xml_parameters.initial_accelerations} file")
        log_accelerations = read_2D_array(xml_parameters.initial_accelerations).reshape((len(dataset.times),))

    else:
        logger.info("Initializing all the onset_ages to the reference time.")
        onset_ages = np.zeros((number_of_subjects,))
        onset_ages += model.get_reference_time()
        logger.info("Initializing all log-accelerations to zero.")
        log_accelerations = np.zeros((number_of_subjects,))

    individual_RER = {}
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Initialization of the spatiotemporal reference frame.
    initialize_spatiotemporal_reference_frame(model, logger, observation_type=model.observation_type)

    # Sources initialization
    if xml_parameters is not None:
        logger.info(f"Setting initial sources from {xml_parameters.initial_sources} file")
        individual_RER['sources'] = read_2D_array(xml_parameters.initial_sources).reshape(len(dataset.times), model.number_of_sources)

    elif model.number_of_sources > 0:
        # Actually here we initialize the sources to almost zero values to avoid renormalization issues (div 0)
        logger.info("Initializing all sources to zero")
        individual_RER['sources'] = np.random.normal(0,0.1,(number_of_subjects, model.number_of_sources))
    model.initialize_source_variables()

    if dataset is not None:
        total_number_of_observations = dataset.total_number_of_observations
        model.number_of_subjects = dataset.number_of_subjects

        if model.get_noise_variance() is None:

            v0, p0, modulation_matrix = model._fixed_effects_to_torch_tensors(False)
            onset_ages, log_accelerations, sources = model._individual_RER_to_torch_tensors(individual_RER, False)

            residuals = model._compute_residuals(dataset, v0, p0, modulation_matrix,
                                            log_accelerations, onset_ages, sources)

            total_residual = 0.
            for i in range(len(residuals)):
                total_residual += torch.sum(residuals[i]).cpu().data.numpy()

            dof = total_number_of_observations
            nv = total_residual / dof
            model.set_noise_variance(nv)
            logger.info(f">> Initial noise variance set to {nv} based on the initial mean residual value.")

        if not model.is_frozen['noise_variance']:
            dof = total_number_of_observations
            model.priors['noise_variance'].degrees_of_freedom.append(dof)

    else:
        if model.get_noise_variance() is None:
            logger.info("I can't initialize the initial noise variance: no dataset and no initialization given.")

    model.update()

    return model, dataset, individual_RER


def estimate_longitudinal_auto_encoder_model(logger, path_data, path_CAE, path_LAE):
    logger.info('')
    logger.info('[ estimate_longitudinal_metric_model function ]')

    torch_data = torch.load(path_data)
    image_data = Dataset(torch_data['data'].unsqueeze(1), torch_data['target'])

    number_of_subjects = len(np.unique([label[0] for label in image_data.labels]))
    model, dataset, individual_RER = instantiate_longitudinal_auto_encoder_model(logger, path_data, path_CAE=path_CAE, path_LAE=path_LAE,
                                                                        number_of_subjects=number_of_subjects)

    sampler = SrwMhwgSampler()
    estimator = McmcSaem(model, dataset, 'McmcSaem', individual_RER, max_iterations=20,
             print_every_n_iters=1, save_every_n_iters=10)
    estimator.sampler = sampler

    # Onset age proposal distribution.
    onset_age_proposal_distribution = MultiScalarNormalDistribution()
    onset_age_proposal_distribution.set_variance_sqrt(5)
    sampler.individual_proposal_distributions['onset_age'] = onset_age_proposal_distribution

    # Log-acceleration proposal distribution.
    log_acceleration_proposal_distribution = MultiScalarNormalDistribution()
    log_acceleration_proposal_distribution.set_variance_sqrt(0.1)
    sampler.individual_proposal_distributions['log_acceleration'] = log_acceleration_proposal_distribution

    # Sources proposal distribution

    if model.number_of_sources > 0:
        sources_proposal_distribution = MultiScalarNormalDistribution()
        # Here we impose the sources variance to be 1
        sources_proposal_distribution.set_variance_sqrt(1)
        sampler.individual_proposal_distributions['sources'] = sources_proposal_distribution

    estimator.sample_every_n_mcmc_iters = 5
    estimator._initialize_acceptance_rate_information()

    # Gradient-based estimator.
    estimator.gradient_based_estimator = GradientAscent(model, dataset, 'GradientAscent', individual_RER)
    estimator.gradient_based_estimator.statistical_model = model
    estimator.gradient_based_estimator.dataset = dataset
    estimator.gradient_based_estimator.optimized_log_likelihood = 'class2'
    estimator.gradient_based_estimator.max_iterations = 40
    estimator.gradient_based_estimator.max_line_search_iterations = 4
    estimator.gradient_based_estimator.convergence_tolerance = 1e-4
    estimator.gradient_based_estimator.print_every_n_iters = 5
    estimator.gradient_based_estimator.save_every_n_iters = 100000
    #estimator.gradient_based_estimator.initial_step_size = 0.1
    estimator.gradient_based_estimator.line_search_shrink = 0.4
    estimator.gradient_based_estimator.line_search_expand = 1.2
    estimator.gradient_based_estimator.scale_initial_step_size = True

    estimator.print_every_n_iters = 1
    estimator.save_every_n_iters = 1

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalMetricModel'
    logger.info('')
    logger.info(f"[ update method of the {estimator.name}  optimizer ]")

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    logger.info(f">> Estimation took: {end_time-start_time}")

def main():
    path_data = 'large_dataset'
    path_CAE = 'CAE'
    path_LAE = 'LAE'
    Settings().dimension = 10
    Settings().number_of_sources = 4
    deformetrica = dfca.Deformetrica(output_dir='output', verbosity=logger.level)
    estimate_longitudinal_auto_encoder_model(logger, path_data, path_CAE, path_LAE)

    print('ok')


if __name__ == "__main__":
    main()