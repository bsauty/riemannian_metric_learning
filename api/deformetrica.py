import gc
import logging
import math
import os
import resource
import sys
import time

import torch
import numpy as np

from core import default, GpuMode
from core.estimators.gradient_ascent import GradientAscent
from core.estimators.mcmc_saem import McmcSaem
from core.estimators.scipy_optimize import ScipyOptimize
from in_out.dataset_functions import create_dataset
from in_out.deformable_object_reader import DeformableObjectReader
from launch.compute_parallel_transport import compute_parallel_transport
from launch.compute_shooting import compute_shooting
from support import utilities
from support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution

global logger
logger = logging.getLogger()


class Deformetrica:
    """ Analysis of 2D and 3D shape data.
    Compute deformations of the 2D or 3D ambient space, which, in turn, warp any object embedded in this space, whether this object is a curve, a surface,
    a structured or unstructured set of points, an image, or any combination of them.
    2 main applications are contained within Deformetrica: `compute` and `estimate`.
    """

    ####################################################################################################################
    # Constructor & destructor.
    ####################################################################################################################

    def __init__(self, output_dir=default.output_dir, verbosity='INFO'):
        """
        Constructor
        :param str output_dir: Path to the output directory
        :param str verbosity: Defines the output log verbosity level. By default the verbosity level is set to 'INFO'.
                          Possible values are: CRITICAL, ERROR, WARNING, INFO or DEBUG

        :raises toto: :py:class:`BaseException`.
        """
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if logger.hasHandlers():
            logger.handlers.clear()

        # file logger
        logger_file_handler = logging.FileHandler(
            os.path.join(self.output_dir, time.strftime("%Y-%m-%d-%H%M%S", time.gmtime()) + '_info.log'), mode='w')
        logger_file_handler.setFormatter(logging.Formatter(default.logger_format))
        logger_file_handler.setLevel(logging.INFO)
        logger.addHandler(logger_file_handler)

        # console logger
        logger_stream_handler = logging.StreamHandler(stream=sys.stdout)
        # logger_stream_handler.setFormatter(logging.Formatter(default.logger_format))
        # logger_stream_handler.setLevel(verbosity)
        try:
            logger_stream_handler.setLevel(verbosity)
            logger.setLevel(verbosity)
        except ValueError:
            logger.warning('Logging level was not recognized. Using INFO.')
            logger_stream_handler.setLevel(logging.INFO)

        logger.addHandler(logger_stream_handler)
        logger.error("Logger has been set to: " + logging.getLevelName(logger_stream_handler.level))

    def __del__(self):
        logger.debug('Deformetrica.__del__()')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # remove previously set env variable
        if 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']

        logging.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('Deformetrica.__exit__()')

    @staticmethod
    def set_seed(seed=None):
        """
        Set the random number generator's seed.
        :param seed: Can be set to None to reset to the original seed
        """
        if seed is None:
            torch.manual_seed(torch.initial_seed())
            np.random.seed(seed)
        else:
            assert isinstance(seed, int)
            torch.manual_seed(seed)
            np.random.seed(seed)

    ####################################################################################################################
    # Main methods.
    ####################################################################################################################


    def compute_parallel_transport(self, template_specifications, model_options={}):
        """ Given a known progression of shapes, to transport this progression onto a new shape.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, _ = self.further_initialization(
            'ParallelTransport', template_specifications, model_options)

        logger.debug("dtype=" + default.dtype)

        # Launch.
        compute_parallel_transport(template_specifications, output_dir=self.output_dir, **model_options)

    def compute_shooting(self, template_specifications, model_options={}):
        """ If control points and momenta corresponding to a deformation have been obtained, 
        it is possible to shoot the corresponding deformation of obtain the flow of a shape under this deformation.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, _ = self.further_initialization(
            'ParallelTransport', template_specifications, model_options)

        logger.debug("dtype=" + default.dtype)

        # Launch.
        compute_shooting(template_specifications, output_dir=self.output_dir, **model_options)


    ####################################################################################################################
    # Auxiliary methods.
    ####################################################################################################################

    @staticmethod
    def __launch_estimator(estimator, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`, :class:`ScipyOptimize <core.estimators.scipy_optimize.ScipyOptimize>`
        """
        logger.debug("dtype=" + default.dtype)
        start_time = time.time()
        logger.info('>> Started estimator: ' + estimator.name)
        estimator.update()
        end_time = time.time()

        if write_output:
            estimator.write()

        if end_time - start_time > 60 * 60 * 24:
            logger.info('>> Estimation took: %s' %
                        time.strftime("%d days, %H hours, %M minutes and %S seconds",
                                      time.gmtime(end_time - start_time)))
        elif end_time - start_time > 60 * 60:
            logger.info('>> Estimation took: %s' %
                        time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(end_time - start_time)))
        elif end_time - start_time > 60:
            logger.info('>> Estimation took: %s' %
                        time.strftime("%M minutes and %S seconds", time.gmtime(end_time - start_time)))
        else:
            logger.info('>> Estimation took: %s' % time.strftime("%S seconds", time.gmtime(end_time - start_time)))

    def __instantiate_estimator(self, statistical_model, dataset, estimator_options, default=ScipyOptimize):
        if estimator_options['optimization_method_type'].lower() == 'GradientAscent'.lower():
            estimator = GradientAscent
        elif estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower():
            estimator = ScipyOptimize
        elif estimator_options['optimization_method_type'].lower() == 'McmcSaem'.lower():
            estimator = McmcSaem
        else:
            estimator = default

        logger.debug(estimator_options)
        return estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

    def further_initialization(self, model_type, template_specifications, model_options,
                               dataset_specifications=None, estimator_options=None):

        #
        # Consistency checks.
        #
        if dataset_specifications is None or estimator_options is None:
            assert model_type.lower() in ['Shooting'.lower(), 'ParallelTransport'.lower()], \
                'Only the "shooting" and "parallel transport" can run without a dataset and an estimator.'

        #
        # Initializes variables that will be checked.
        #
        if estimator_options is not None:
            if 'gpu_mode' not in estimator_options:
                estimator_options['gpu_mode'] = default.gpu_mode
            if estimator_options['gpu_mode'] is GpuMode.FULL and not torch.cuda.is_available():
                logger.warning("GPU computation is not available, falling-back to CPU.")
                estimator_options['gpu_mode'] = GpuMode.NONE

            if 'state_file' not in estimator_options:
                estimator_options['state_file'] = default.state_file
            if 'load_state_file' not in estimator_options:
                estimator_options['load_state_file'] = default.load_state_file
            if 'memory_length' not in estimator_options:
                estimator_options['memory_length'] = default.memory_length

        if 'dimension' not in model_options:
            model_options['dimension'] = default.dimension
        if 'dtype' not in model_options:
            model_options['dtype'] = default.dtype
        else:
            default.update_dtype(new_dtype=model_options['dtype'])

        model_options['tensor_scalar_type'] = default.tensor_scalar_type
        model_options['tensor_integer_type'] = default.tensor_integer_type

        if 'dense_mode' not in model_options:
            model_options['dense_mode'] = default.dense_mode
        if 'freeze_control_points' not in model_options:
            model_options['freeze_control_points'] = default.freeze_control_points
        if 'freeze_template' not in model_options:
            model_options['freeze_template'] = default.freeze_template
        if 'initial_control_points' not in model_options:
            model_options['initial_control_points'] = default.initial_control_points
        if 'initial_cp_spacing' not in model_options:
            model_options['initial_cp_spacing'] = default.initial_cp_spacing
        if 'deformation_kernel_width' not in model_options:
            model_options['deformation_kernel_width'] = default.deformation_kernel_width
        if 'deformation_kernel_type' not in model_options:
            model_options['deformation_kernel_type'] = default.deformation_kernel_type
        if 'number_of_processes' not in model_options:
            model_options['number_of_processes'] = default.number_of_processes
        if 't0' not in model_options:
            model_options['t0'] = default.t0
        if 'initial_time_shift_variance' not in model_options:
            model_options['initial_time_shift_variance'] = default.initial_time_shift_variance
        if 'initial_modulation_matrix' not in model_options:
            model_options['initial_modulation_matrix'] = default.initial_modulation_matrix
        if 'number_of_sources' not in model_options:
            model_options['number_of_sources'] = default.number_of_sources
        if 'initial_acceleration_variance' not in model_options:
            model_options['initial_acceleration_variance'] = default.initial_acceleration_variance
        if 'downsampling_factor' not in model_options:
            model_options['downsampling_factor'] = default.downsampling_factor
        if 'use_sobolev_gradient' not in model_options:
            model_options['use_sobolev_gradient'] = default.use_sobolev_gradient
        if 'sobolev_kernel_width_ratio' not in model_options:
            model_options['sobolev_kernel_width_ratio'] = default.sobolev_kernel_width_ratio

        #
        # Check and completes the user-given parameters.
        #

        # Optional random seed.
        if 'random_seed' in model_options and model_options['random_seed'] is not None:
            self.set_seed(model_options['random_seed'])

        # If needed, infer the dimension from the template specifications.
        if model_options['dimension'] is None:
            model_options['dimension'] = self.__infer_dimension(template_specifications)

        # Smoothing kernel width.
        if model_options['use_sobolev_gradient']:
            model_options['smoothing_kernel_width'] = \
                model_options['deformation_kernel_width'] * model_options['sobolev_kernel_width_ratio']

        # Dense mode.
        if model_options['dense_mode']:
            logger.info('>> Dense mode activated. No distinction will be made between template and control points.')
            assert len(template_specifications) == 1, \
                'Only a single object can be considered when using the dense mode.'
            if not model_options['freeze_control_points']:
                model_options['freeze_control_points'] = True
                msg = 'With active dense mode, the freeze_template (currently %s) and freeze_control_points ' \
                      '(currently %s) flags are redundant. Defaulting to freeze_control_points = True.' \
                      % (str(model_options['freeze_template']), str(model_options['freeze_control_points']))
                logger.info('>> ' + msg)
            if model_options['initial_control_points'] is not None:
                # model_options['initial_control_points'] = None
                msg = 'With active dense mode, specifying initial_control_points is useless. Ignoring this xml entry.'
                logger.info('>> ' + msg)

        if model_options['initial_cp_spacing'] is None and model_options['initial_control_points'] is None \
                and not model_options['dense_mode']:
            logger.info('>> No initial CP spacing given: using diffeo kernel width of '
                        + str(model_options['deformation_kernel_width']))
            model_options['initial_cp_spacing'] = model_options['deformation_kernel_width']

        # Multi-threading/processing only available for the deterministic atlas for the moment.
        if model_options['number_of_processes'] > 1:

            if model_type.lower() in ['Shooting'.lower(), 'ParallelTransport'.lower(), 'Registration'.lower()]:
                model_options['number_of_processes'] = 1
                msg = 'It is not possible to estimate a "%s" model with multithreading. ' \
                      'Overriding the "number-of-processes" option, now set to 1.' % model_type
                logger.info('>> ' + msg)

            elif model_type.lower() in ['BayesianAtlas'.lower(), 'Regression'.lower(),
                                        'LongitudinalRegistration'.lower()]:
                model_options['number_of_processes'] = 1
                msg = 'It is not possible at the moment to estimate a "%s" model with multithreading. ' \
                      'Overriding the "number-of-processes" option, now set to 1.' % model_type
                logger.info('>> ' + msg)

        # try and automatically set best number of thread per spawned process if not overridden by uer
        if 'OMP_NUM_THREADS' not in os.environ:
            logger.info('OMP_NUM_THREADS was not found in environment variables. An automatic value will be set.')
            hyperthreading = utilities.has_hyperthreading()
            omp_num_threads = math.floor(os.cpu_count() / model_options['number_of_processes'])

            if hyperthreading:
                omp_num_threads = math.ceil(omp_num_threads / 2)

            omp_num_threads = max(1, int(omp_num_threads))

            logger.info('OMP_NUM_THREADS will be set to ' + str(omp_num_threads))
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
            # os.environ['OMP_PLACES'] = 'sockets'    # threads, cores, sockets, {...}
            # os.environ['OMP_PROC_BIND'] = 'close'  # close, spread, master
        else:
            logger.info('OMP_NUM_THREADS found in environment variables. Using value OMP_NUM_THREADS=' + str(
                os.environ['OMP_NUM_THREADS']))

        # If longitudinal model and t0 is not initialized, initializes it.
        if model_type.lower() in ['Regression'.lower(),
                                  'LongitudinalAtlas'.lower(), 'LongitudinalRegistration'.lower()]:
            total_number_of_visits = 0
            mean_visit_age = 0.0
            var_visit_age = 0.0
            assert 'visit_ages' in dataset_specifications, 'Visit ages are needed to estimate a Regression, ' \
                                                           'Longitudinal Atlas or Longitudinal Registration model.'
            for i in range(len(dataset_specifications['visit_ages'])):
                for j in range(len(dataset_specifications['visit_ages'][i])):
                    total_number_of_visits += 1
                    mean_visit_age += dataset_specifications['visit_ages'][i][j]
                    var_visit_age += dataset_specifications['visit_ages'][i][j] ** 2

            if total_number_of_visits > 0:
                mean_visit_age /= float(total_number_of_visits)
                var_visit_age = (var_visit_age / float(total_number_of_visits) - mean_visit_age ** 2)

                if model_options['t0'] is None:
                    logger.info('>> Initial t0 set to the mean visit age: %.2f' % mean_visit_age)
                    model_options['t0'] = mean_visit_age
                else:
                    logger.info('>> Initial t0 set by the user to %.2f ; note that the mean visit age is %.2f'
                                % (model_options['t0'], mean_visit_age))

                if not model_type.lower() == 'regression':
                    if model_options['initial_time_shift_variance'] is None:
                        logger.info('>> Initial time-shift std set to the empirical std of the visit ages: %.2f'
                                    % math.sqrt(var_visit_age))
                        model_options['initial_time_shift_variance'] = var_visit_age
                    else:
                        logger.info(
                            ('>> Initial time-shift std set by the user to %.2f ; note that the empirical std of '
                             'the visit ages is %.2f') % (math.sqrt(model_options['initial_time_shift_variance']),
                                                          math.sqrt(var_visit_age)))

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            # cf: https://discuss.pytorch.org/t/a-call-to-torch-cuda-is-available-makes-an-unrelated-multi-processing-computation-crash/4075/2?u=smth
            torch.multiprocessing.set_start_method("spawn")
            # cf: https://github.com/pytorch/pytorch/issues/11201
            # torch.multiprocessing.set_sharing_strategy('file_system')
            torch.multiprocessing.set_sharing_strategy('file_descriptor')
            # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
            logger.debug("nofile (soft): " + str(rlimit[0]) + ", nofile (hard): " + str(rlimit[1]))
            resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
        except RuntimeError as e:
            logger.warning(str(e))
        except AssertionError:
            logger.warning('Could not set torch settings.')
        except ValueError:
            logger.warning('Could not set max open file. Currently using: ' + str(rlimit))

        if estimator_options is not None:
            # Initializes the state file.
            if estimator_options['state_file'] is None:
                path_to_state_file = os.path.join(self.output_dir, "deformetrica-state.p")
                logger.info('>> No specified state-file. By default, Deformetrica state will by saved in file: %s.' %
                            path_to_state_file)
                if os.path.isfile(path_to_state_file):
                    os.remove(path_to_state_file)
                    logger.info('>> Removing the pre-existing state file with same path.')
                estimator_options['state_file'] = path_to_state_file
            else:
                if os.path.exists(estimator_options['state_file']):
                    estimator_options['load_state_file'] = True
                    logger.info(
                        '>> Deformetrica will attempt to resume computation from the user-specified state file: %s.'
                        % estimator_options['state_file'])
                else:
                    msg = 'The user-specified state-file does not exist: %s. State cannot be reloaded. ' \
                          'Future Deformetrica state will be saved at the given path.' % estimator_options['state_file']
                    logger.info('>> ' + msg)

            # Warning if scipy-LBFGS with memory length > 1 and sobolev gradient.
            if estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower() \
                    and estimator_options['memory_length'] > 1 \
                    and not model_options['freeze_template'] and model_options['use_sobolev_gradient']:
                logger.info(
                    '>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                    'being larger than 1. Beware: that can be tricky.')

        # Checking the number of image objects, and moving as desired the downsampling_factor parameter.
        count = 0
        for elt in template_specifications.values():
            if elt['deformable_object_type'].lower() == 'image':
                count += 1
                if not model_options['downsampling_factor'] == 1:
                    if 'downsampling_factor' in elt.keys():
                        logger.info('>> Warning: the downsampling_factor option is specified twice. '
                                    'Taking the value: %d.' % elt['downsampling_factor'])
                    else:
                        elt['downsampling_factor'] = model_options['downsampling_factor']
                        logger.info('>> Setting the image grid downsampling factor to: %d.' %
                                    model_options['downsampling_factor'])
        if count > 1:
            raise RuntimeError('Only a single image object can be used.')
        if count == 0 and not model_options['downsampling_factor'] == 1:
            msg = 'The "downsampling_factor" parameter is useful only for image data, ' \
                  'but none is considered here. Ignoring.'
            logger.info('>> ' + msg)

        # Initializes the proposal distributions.
        if estimator_options is not None and \
                estimator_options['optimization_method_type'].lower() == 'McmcSaem'.lower():

            assert model_type.lower() in ['LongitudinalAtlas'.lower(), 'BayesianAtlas'.lower()], \
                'Only the "BayesianAtlas" and "LongitudinalAtlas" models can be estimated with the "McmcSaem" ' \
                'algorithm, when here was specified a "%s" model.' % model_type

            if model_type.lower() == 'LongitudinalAtlas'.lower():

                if 'onset_age_proposal_std' not in estimator_options:
                    estimator_options['onset_age_proposal_std'] = default.onset_age_proposal_std
                if 'acceleration_proposal_std' not in estimator_options:
                    estimator_options['acceleration_proposal_std'] = default.acceleration_proposal_std
                if 'sources_proposal_std' not in estimator_options:
                    estimator_options['sources_proposal_std'] = default.sources_proposal_std

                estimator_options['individual_proposal_distributions'] = {
                    'onset_age': MultiScalarNormalDistribution(std=estimator_options['onset_age_proposal_std']),
                    'acceleration': MultiScalarNormalDistribution(std=estimator_options['acceleration_proposal_std']),
                    'sources': MultiScalarNormalDistribution(std=estimator_options['sources_proposal_std'])}

            elif model_type.lower() == 'BayesianAtlas'.lower():
                if 'momenta_proposal_std' not in estimator_options:
                    estimator_options['momenta_proposal_std'] = default.momenta_proposal_std

                estimator_options['individual_proposal_distributions'] = {
                    'momenta': MultiScalarNormalDistribution(std=estimator_options['momenta_proposal_std'])}

        return template_specifications, model_options, estimator_options

    @staticmethod
    def __infer_dimension(template_specifications):
        reader = DeformableObjectReader()
        max_dimension = 0
        for elt in template_specifications.values():
            object_filename = elt['filename']
            object_type = elt['deformable_object_type']
            o = reader.create_object(object_filename, object_type, dimension=None)
            d = o.dimension
            max_dimension = max(d, max_dimension)
        return max_dimension
