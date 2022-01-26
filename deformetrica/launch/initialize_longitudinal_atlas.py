import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from torch.autograd import Variable

import warnings
from decimal import Decimal
import shutil
import math
from sklearn.decomposition import PCA, FastICA
import torch
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from scipy.stats import norm, truncnorm

from ..core import default
from ..in_out.xml_parameters import XmlParameters, get_dataset_specifications, get_estimator_options, get_model_options
from ..in_out.dataset_functions import create_template_metadata
from ..core.model_tools.deformations.exponential import Exponential
from ..core.model_tools.deformations.geodesic import Geodesic
from ..in_out.array_readers_and_writers import *
from ..support import kernels as kernel_factory
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import DeformableObjectReader
from ..api.deformetrica import Deformetrica


def estimate_bayesian_atlas(deformetrica, xml_parameters):
    model, individual_RER = deformetrica.estimate_bayesian_atlas(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters))
    return model, individual_RER['momenta']


def estimate_deterministic_atlas(deformetrica, xml_parameters):
    return deformetrica.estimate_deterministic_atlas(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters))


def estimate_geodesic_regression(deformetrica, xml_parameters):
    return deformetrica.estimate_geodesic_regression(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters))


def estimate_longitudinal_registration(deformetrica, xml_parameters, overwrite=True):
    return deformetrica.estimate_longitudinal_registration(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters), overwrite=overwrite)


def estimate_longitudinal_atlas(deformetrica, xml_parameters):
    return deformetrica.estimate_longitudinal_atlas(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters))


def insert_model_xml_level1_entry(model_xml_level0, key, value):
    found_tag = False
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == key:
            model_xml_level1.text = value
            found_tag = True
    if not found_tag:
        new_element_xml = et.SubElement(model_xml_level0, key)
        new_element_xml.text = value
    return model_xml_level0


def insert_model_xml_template_spec_entry(model_xml_level0, key, values):
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'template':
            k = -1
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == 'object':
                    k += 1
                    found_tag = False
                    for model_xml_level3 in model_xml_level2:
                        if model_xml_level3.tag.lower() == key.lower():
                            model_xml_level3.text = values[k]
                            found_tag = True
                    if not found_tag:
                        new_element_xml = et.SubElement(model_xml_level2, key)
                        new_element_xml.text = values[k]
    return model_xml_level0


def insert_model_xml_deformation_parameters_entry(model_xml_level0, key, value):
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'deformation-parameters':
            found_tag = False
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == key:
                    model_xml_level2.text = value
                    found_tag = True
            if not found_tag:
                new_element_xml = et.SubElement(model_xml_level1, key)
                new_element_xml.text = value
    return model_xml_level0


def estimate_geodesic_regression_for_subject(
        i, deformetrica, xml_parameters, regressions_output_path,
        global_full_dataset_filenames, global_full_visit_ages, global_full_subject_ids):
    logger.info('')
    logger.info('[ geodesic regression for subject ' + global_full_subject_ids[i] + ' ]')
    logger.info('')

    # Create folder.
    subject_regression_output_path = os.path.join(regressions_output_path,
                                                  'GeodesicRegression__subject_' + global_full_subject_ids[i])
    if os.path.isdir(subject_regression_output_path): shutil.rmtree(subject_regression_output_path)
    os.mkdir(subject_regression_output_path)

    # Adapt the specific xml parameters and update.Logger has been set to
    xml_parameters.dataset_filenames = [global_full_dataset_filenames[i]]
    xml_parameters.visit_ages = [global_full_visit_ages[i]]
    xml_parameters.subject_ids = [global_full_subject_ids[i]]
    xml_parameters.t0 = xml_parameters.visit_ages[0][0]
    xml_parameters.state_file = None

    # Adapt the global settings, for the custom output directory.
    deformetrica.output_dir = subject_regression_output_path
    # Settings().state_file = os.path.join(Settings().output_dir, 'pydef_state.p')
    # xml_parameters._further_initialization(deformetrica.output_dir)

    # Launch.
    model = estimate_geodesic_regression(deformetrica, xml_parameters)

    # Add the estimated momenta.
    return model.get_control_points(), model.get_momenta()


def shoot(control_points, momenta, kernel_width, kernel_type, kernel_device,
          number_of_time_points=default.number_of_time_points,
          dense_mode=default.dense_mode,
          tensor_scalar_type=default.tensor_scalar_type):
    control_points_torch = torch.from_numpy(control_points).type(tensor_scalar_type)
    momenta_torch = torch.from_numpy(momenta).type(tensor_scalar_type)
    exponential = Exponential(
        dense_mode=dense_mode,
        kernel=kernel_factory.factory(kernel_type, kernel_width=kernel_width),
        number_of_time_points=number_of_time_points,
        initial_control_points=control_points_torch, initial_momenta=momenta_torch)
    exponential.shoot()
    return exponential.control_points_t[-1].detach().cpu().numpy(), exponential.momenta_t[-1].detach().cpu().numpy()


def reproject_momenta(source_control_points, source_momenta, target_control_points,
                      kernel_width, kernel_type='torch', kernel_device='cpu',
                      tensor_scalar_type=default.tensor_scalar_type):
    kernel = kernel_factory.factory(kernel_type, kernel_width=kernel_width)
    source_control_points_torch = tensor_scalar_type(source_control_points)
    source_momenta_torch = tensor_scalar_type(source_momenta)
    target_control_points_torch = tensor_scalar_type(target_control_points)
    target_momenta_torch = torch.cholesky_solve(
        kernel.convolve(target_control_points_torch, source_control_points_torch, source_momenta_torch),
        torch.cholesky(kernel.get_kernel_matrix(target_control_points_torch), upper=True), upper=True)
    # target_momenta_torch_bis = torch.mm(torch.inverse(kernel.get_kernel_matrix(target_control_points_torch)),
    #                                     kernel.convolve(target_control_points_torch, source_control_points_torch,
    #                                                     source_momenta_torch))
    return target_momenta_torch.detach().cpu().numpy()


def parallel_transport(source_control_points, source_momenta, driving_momenta,
                       kernel_width, kernel_type='torch', kernel_device='cpu',
                       number_of_time_points=default.number_of_time_points,
                       dense_mode=default.dense_mode,
                       tensor_scalar_type=default.tensor_scalar_type):
    source_control_points_torch = tensor_scalar_type(source_control_points)
    source_momenta_torch = tensor_scalar_type(source_momenta)
    driving_momenta_torch = tensor_scalar_type(driving_momenta)
    exponential = Exponential(
        dense_mode=dense_mode,
        kernel=kernel_factory.factory(kernel_type, kernel_width=kernel_width),
        number_of_time_points=number_of_time_points,
        use_rk2_for_shoot=True,
        initial_control_points=source_control_points_torch, initial_momenta=driving_momenta_torch)
    exponential.shoot()
    transported_control_points_torch = exponential.control_points_t[-1]
    transported_momenta_torch = exponential.parallel_transport(source_momenta_torch)[-1]
    return transported_control_points_torch.detach().cpu().numpy(), transported_momenta_torch.detach().cpu().numpy()


def initialize_longitudinal_atlas(model_xml_path, dataset_xml_path, optimization_parameters_xml_path,
                                  output_dir='preprocessing_3', overwrite=False):
    """
    0]. Read command line, change directory, prepare preprocessing_3 folder, read original xml parameters.
    """

    # assert len(sys.argv) >= 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> " \
    #                                                      "<optional --overwrite>"

    # model_xml_path = sys.argv[1]
    # dataset_xml_path = sys.argv[2]
    # optimization_parameters_xml_path = sys.argv[3]

    preprocessings_folder = output_dir
    preprocessings_folder_exists = os.path.isdir(preprocessings_folder)
    if not preprocessings_folder_exists:
        os.mkdir(preprocessings_folder)

    global_path_to_data = os.path.join(os.path.dirname(preprocessings_folder), 'data')
    if not os.path.isdir(global_path_to_data):
        os.mkdir(global_path_to_data)

    global_overwrite = True
    # global_overwrite = False
    # if overwrite and preprocessings_folder_exists:
    #     logger.info('>> The script will overwrite the results from already performed initialization steps.')
    #     user_answer = input('>> Proceed with overwriting ? ([y]es / [n]o)')
    #     if str(user_answer).lower() in ['y', 'yes']:
    #         global_overwrite = True
    #     elif not str(user_answer).lower() in ['n', 'no']:
    #         logger.info('>> Unexpected answer. Proceeding without overwriting.')

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    global_deformetrica = Deformetrica(output_dir=preprocessings_folder)
    template_specifications, model_options, estimator_options = global_deformetrica.further_initialization(
        'LongitudinalAtlas', xml_parameters.template_specifications, get_model_options(xml_parameters),
        dataset_specifications=get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters))
    # os.rmdir('tmp')

    # Save some global parameters.
    global_full_dataset_filenames = xml_parameters.dataset_filenames
    global_full_visit_ages = xml_parameters.visit_ages
    global_full_subject_ids = xml_parameters.subject_ids

    global_objects_name, global_objects_name_extension \
        = create_template_metadata(xml_parameters.template_specifications)[1:3]

    global_user_specified_optimization_method = xml_parameters.optimization_method_type
    global_user_specified_number_of_processes = xml_parameters.number_of_processes

    global_dense_mode = xml_parameters.dense_mode
    global_deformation_kernel_type = xml_parameters.deformation_kernel_type
    global_deformation_kernel_width = xml_parameters.deformation_kernel_width
    global_deformation_kernel_device = xml_parameters.deformation_kernel_device

    global_number_of_subjects = len(global_full_dataset_filenames)
    global_total_number_of_observations = sum([len(elt) for elt in global_full_visit_ages])

    global_initial_control_points_are_given = xml_parameters.initial_control_points is not None
    global_initial_momenta_are_given = xml_parameters.initial_momenta is not None
    global_initial_modulation_matrix_is_given = xml_parameters.initial_modulation_matrix is not None
    global_initial_t0_is_given = xml_parameters.t0 is not None

    global_t0 = xml_parameters.t0
    if not global_initial_t0_is_given:
        global_t0 = sum([sum(elt) for elt in global_full_visit_ages]) / float(global_total_number_of_observations)

    global_tmin = sum([elt[0] for elt in global_full_visit_ages]) / float(global_number_of_subjects)

    global_number_of_time_points = xml_parameters.number_of_time_points

    global_tensor_scalar_type = model_options['tensor_scalar_type']
    global_tensor_integer_type = model_options['tensor_integer_type']

    """
    1]. Compute an atlas on the baseline data.
    ------------------------------------------
        The outputted template, control points and noise standard deviation will be used in the following
        geodesic regression and longitudinal registration, as well as an initialization for the longitudinal atlas.
        The template will always be used, i.e. the user-provided one is assumed to be a dummy, low-quality one.
        On the other hand, the estimated control points and noise standard deviation will only be used if the user did
        not provide those.
    """

    atlas_type = 'Bayesian'
    # atlas_type = 'Deterministic'

    atlas_output_path = os.path.join(preprocessings_folder, '1_atlas_on_baseline_data')
    if not global_overwrite and os.path.isdir(atlas_output_path):

        global_initial_control_points = read_2D_array(os.path.join(
            global_path_to_data, 'ForInitialization__ControlPoints__FromAtlas.txt'))
        global_atlas_momenta = read_3D_array(os.path.join(
            atlas_output_path, atlas_type + 'Atlas__EstimatedParameters__Momenta.txt'))
        global_dimension = global_initial_control_points.shape[1]

        global_initial_objects_template_list = []
        global_initial_objects_template_path = []
        global_initial_objects_template_type = []
        reader = DeformableObjectReader()
        for object_id, object_specs in xml_parameters.template_specifications.items():
            extension = os.path.splitext(object_specs['filename'])[-1]
            filename = os.path.join(global_path_to_data, 'ForInitialization__Template_%s__FromAtlas%s' % (object_id, extension))
            object_type = object_specs['deformable_object_type'].lower()
            template_object = reader.create_object(filename, object_type)
            global_initial_objects_template_list.append(template_object)
            global_initial_objects_template_path.append(filename)
            global_initial_objects_template_type.append(template_object.type.lower())

        global_initial_template = DeformableMultiObject(global_initial_objects_template_list)
        global_initial_template.update()

        global_initial_template_data = global_initial_template.get_data()

        model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')

    else:
        logger.info('[ estimate an atlas from baseline data ]')
        logger.info('')

        # Initialization -----------------------------------------------------------------------------------------------
        # Clean folder.
        if os.path.isdir(atlas_output_path): shutil.rmtree(atlas_output_path)
        os.mkdir(atlas_output_path)

        # Adapt the xml parameters and update.
        xml_parameters.model_type = (atlas_type + 'Atlas').lower()
        # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
        xml_parameters.optimization_method_type = 'GradientAscent'.lower()
        xml_parameters.max_line_search_iterations = 20

        xml_parameters.number_of_processes = 1  # Problem to fix here. TODO.

        # if True or xml_parameters.use_cuda:
        #     xml_parameters.number_of_processes = 1  # Problem to fix here. TODO.
        #     global_user_specified_number_of_processes = 1
        xml_parameters.print_every_n_iters = 1

        xml_parameters.initial_momenta = None

        xml_parameters.dataset_filenames = [[elt[0]] for elt in xml_parameters.dataset_filenames]
        xml_parameters.visit_ages = [[elt[0]] for elt in xml_parameters.visit_ages]

        # Adapt the global settings, for the custom output directory.
        global_deformetrica.output_dir = atlas_output_path
        # Settings().state_file = os.path.join(Settings().output_dir, 'pydef_state.p')

        # xml_parameters._further_initialization(global_deformetrica.output_dir)

        # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
        if atlas_type == 'Bayesian':
            model, global_atlas_momenta = estimate_bayesian_atlas(global_deformetrica, xml_parameters)
            global_objects_noise_std = [math.sqrt(elt) for elt in model.get_noise_variance()]

        elif atlas_type == 'Deterministic':
            model = estimate_deterministic_atlas(global_deformetrica, xml_parameters)
            global_objects_noise_std = [math.sqrt(elt) for elt in model.objects_noise_variance]
            global_atlas_momenta = model.get_momenta()

        else:
            raise RuntimeError('Unknown atlas type: "' + atlas_type + '"')
        global_dimension = model.fixed_effects['control_points'].shape[1]

        # Export the results -------------------------------------------------------------------------------------------
        global_objects_name, global_objects_name_extension, original_objects_noise_variance = \
            create_template_metadata(xml_parameters.template_specifications)[1:4]

        global_initial_template = model.template
        global_initial_template_data = model.get_template_data()
        global_initial_control_points = model.get_control_points()

        global_initial_objects_template_path = []
        global_initial_objects_template_type = []
        for k, (object_name, object_name_extension, original_object_noise_variance) \
                in enumerate(zip(global_objects_name, global_objects_name_extension, original_objects_noise_variance)):

            # Save the template objects type.
            global_initial_objects_template_type.append(global_initial_template.object_list[k].type.lower())

            # Copy the estimated template to the data folder.
            estimated_template_path = os.path.join(
                atlas_output_path,
                atlas_type + 'Atlas__EstimatedParameters__Template_' + object_name + object_name_extension)
            global_initial_objects_template_path.append(os.path.join(
                global_path_to_data, 'ForInitialization__Template_' + object_name + '__FromAtlas' + object_name_extension))
            shutil.copyfile(estimated_template_path, global_initial_objects_template_path[k])

            if global_initial_objects_template_type[k] == 'PolyLine'.lower():
                cmd = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
                os.system(cmd)  # Quite time-consuming.
                if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                    os.remove(global_initial_objects_template_path[k] + '--')

            # Override the obtained noise standard deviation values, if it was already given by the user.
            if original_object_noise_variance > 0:
                global_objects_noise_std[k] = math.sqrt(original_object_noise_variance)

        # Convert the noise std float values to formatted strings.
        global_objects_noise_std_string = ['{:.4f}'.format(elt) for elt in global_objects_noise_std]

        # If necessary, copy the estimated control points to the data folder.
        if global_initial_control_points_are_given:
            global_initial_control_points_path = xml_parameters.initial_control_points
            global_initial_control_points = read_2D_array(global_initial_control_points_path)
        else:
            estimated_control_points_path = os.path.join(atlas_output_path,
                                                         atlas_type + 'Atlas__EstimatedParameters__ControlPoints.txt')
            global_initial_control_points_path = os.path.join(global_path_to_data, 'ForInitialization__ControlPoints__FromAtlas.txt')
            shutil.copyfile(estimated_control_points_path, global_initial_control_points_path)

        # Modify and write the model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_template_spec_entry(model_xml_level0,
                                                                'filename', global_initial_objects_template_path)
        model_xml_level0 = insert_model_xml_template_spec_entry(model_xml_level0,
                                                                'noise-std', global_objects_noise_std_string)
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-control-points', global_initial_control_points_path)
        model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
        doc = parseString((et.tostring(
            model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    2]. Compute individual geodesic regressions.
    --------------------------------------------
        The time t0 is chosen as the baseline age for every subject.
        The control points are the one outputted by the atlas estimation, and are frozen.
        Skipped if an initial control points and (longitudinal) momenta are specified.
    """

    # Read the current model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Check if the computations have been done already.
    regressions_output_path = os.path.join(preprocessings_folder, '2_individual_geodesic_regressions')
    if not global_overwrite and os.path.isdir(regressions_output_path):
        global_initial_momenta = read_3D_array(os.path.join(global_path_to_data, 'ForInitialization__Momenta__FromRegressions.txt'))

    # Check if an initial (longitudinal) momenta is available.
    elif global_initial_control_points_are_given and global_initial_momenta_are_given:
        global_initial_momenta = read_3D_array(xml_parameters.initial_momenta)

    else:
        logger.info('')
        logger.info('[ compute individual geodesic regressions ]')

        # Warning.
        if not global_initial_control_points_are_given and global_initial_momenta_are_given:
            msg = 'Initial momenta are given but not the corresponding initial control points. ' \
                  'Those given initial momenta will be ignored, and overridden by a regression-based heuristic.'
            warnings.warn(msg)

        # Create folder.
        if os.path.isdir(regressions_output_path): shutil.rmtree(regressions_output_path)
        os.mkdir(regressions_output_path)

        regression_tmp_path = os.path.join(regressions_output_path, 'tmp')
        if os.path.isdir(regression_tmp_path):
            shutil.rmtree(regression_tmp_path)
        os.mkdir(regression_tmp_path)

        # Adapt the shared xml parameters.
        xml_parameters.model_type = 'Regression'.lower()
        # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
        xml_parameters.optimization_method_type = 'GradientAscent'.lower()
        xml_parameters.max_line_search_iterations = 10
        xml_parameters.initial_control_points = None
        xml_parameters.freeze_control_points = True
        xml_parameters.freeze_template = True
        xml_parameters.print_every_n_iters = 1

        # Launch -------------------------------------------------------------------------------------------------------
        global_initial_momenta = np.zeros(global_initial_control_points.shape)
        for i in range(global_number_of_subjects):

            # Set the initial template as the reconstructed object after the atlas.
            for k, (object_id, object_specs) in enumerate(xml_parameters.template_specifications.items()):
                object_specs['filename'] = os.path.join(
                    atlas_output_path,
                    '%sAtlas__Reconstruction__%s__subject_%s%s' %
                    (atlas_type, object_id, global_full_subject_ids[i], global_objects_name_extension[k]))

            # Find the control points and momenta that transforms the previously computed template into the individual.
            registration_control_points, registration_momenta = shoot(
                global_initial_control_points, global_atlas_momenta[i],
                global_deformation_kernel_width, global_deformation_kernel_type, global_deformation_kernel_device,
                number_of_time_points=global_number_of_time_points,
                dense_mode=global_dense_mode, tensor_scalar_type=global_tensor_scalar_type)

            # Dump those control points, and use them for the regression.
            path_to_regression_control_points = os.path.join(
                regression_tmp_path, 'regression_control_points__%s.txt' % global_full_subject_ids[i])
            np.savetxt(path_to_regression_control_points, registration_control_points)
            xml_parameters.initial_control_points = path_to_regression_control_points

            # Regression.
            regression_control_points, regression_momenta = estimate_geodesic_regression_for_subject(
                i, global_deformetrica, xml_parameters, regressions_output_path,
                global_full_dataset_filenames, global_full_visit_ages, global_full_subject_ids)

            # Parallel transport of the estimated momenta.
            transported_regression_control_points, transported_regression_momenta = parallel_transport(
                regression_control_points, regression_momenta, - registration_momenta,
                global_deformation_kernel_width, global_deformation_kernel_type, global_deformation_kernel_device,
                number_of_time_points=global_number_of_time_points,
                dense_mode=global_dense_mode, tensor_scalar_type=global_tensor_scalar_type)

            # Increment the global initial momenta.
            global_initial_momenta += transported_regression_momenta

            # Saving this transported momenta.
            path_to_subject_transported_regression_momenta = os.path.join(
                regressions_output_path, 'GeodesicRegression__subject_' + global_full_subject_ids[i],
                'GeodesicRegression__EstimatedParameters__TransportedMomenta.txt')
            np.savetxt(path_to_subject_transported_regression_momenta, transported_regression_momenta)

        # Divide to obtain the average momenta. Write the result in the data folder.
        global_initial_momenta /= float(global_number_of_subjects)
        global_initial_momenta_path = os.path.join(global_path_to_data, 'ForInitialization__Momenta__FromRegressions.txt')
        np.savetxt(global_initial_momenta_path, global_initial_momenta)

        # Modify and write the model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-momenta', global_initial_momenta_path)
        model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    3]. Initializing heuristics for accelerations and onset ages.
    -----------------------------------------------------------------
        The individual accelerations are taken as the ratio of the regression momenta norm to the global one.
        The individual onset ages are computed as if all baseline ages were in correspondence.
    """

    logger.info('')
    logger.info('[ initializing heuristics for individual accelerations and onset ages ]')
    logger.info('')

    kernel = kernel_factory.factory('torch',
                                    kernel_width=xml_parameters.deformation_kernel_width)

    global_initial_control_points_torch = torch.from_numpy(
        global_initial_control_points).type(global_tensor_scalar_type)

    global_initial_momenta_torch = torch.from_numpy(global_initial_momenta).type(global_tensor_scalar_type)
    global_initial_momenta_norm_squared = torch.dot(global_initial_momenta_torch.view(-1), kernel.convolve(
        global_initial_control_points_torch, global_initial_control_points_torch,
        global_initial_momenta_torch).view(-1)).detach().cpu().numpy()

    heuristic_initial_onset_ages = []
    heuristic_initial_accelerations = []
    for i in range(global_number_of_subjects):

        # Heuristic for the initial onset age.
        subject_mean_observation_age = np.mean(np.array(global_full_visit_ages[i]))
        heuristic_initial_onset_ages.append(subject_mean_observation_age)

        # Heuristic for the initial acceleration.
        path_to_subject_transported_regression_momenta = os.path.join(
            regressions_output_path, 'GeodesicRegression__subject_' + global_full_subject_ids[i],
            'GeodesicRegression__EstimatedParameters__TransportedMomenta.txt')
        subject_regression_momenta = read_3D_array(path_to_subject_transported_regression_momenta)
        subject_regression_momenta_torch = torch.from_numpy(
            subject_regression_momenta).type(global_tensor_scalar_type)

        subject_regression_momenta_scalar_product_with_population_momenta = torch.dot(
            global_initial_momenta_torch.view(-1), kernel.convolve(
                global_initial_control_points_torch, global_initial_control_points_torch,
                subject_regression_momenta_torch).view(-1)).detach().cpu().numpy()

        if subject_regression_momenta_scalar_product_with_population_momenta <= 0.0:
            msg = 'Subject %s seems to evolve against the population: scalar_product = %.3E.' % \
                  (global_full_subject_ids[i],
                   Decimal(float(subject_regression_momenta_scalar_product_with_population_momenta)))
            warnings.warn(msg)
            logger.info('>> ' + msg)
            heuristic_initial_accelerations.append(1.0)  # Neutral initialization.
        else:
            heuristic_initial_accelerations.append(
                float(np.sqrt(subject_regression_momenta_scalar_product_with_population_momenta
                              / global_initial_momenta_norm_squared)))

    heuristic_initial_onset_ages = np.array(heuristic_initial_onset_ages)
    heuristic_initial_accelerations = np.array(heuristic_initial_accelerations)

    def get_acceleration_std_from_accelerations(accelerations):  # Fixed-point algorithm.
        ss = np.sum((accelerations - 1.0) ** 2)
        number_of_subjects = len(accelerations)
        max_number_of_iterations = 100
        convergence_tolerance = 1e-5
        std_old, std_new = math.sqrt(ss / float(number_of_subjects)), math.sqrt(ss / float(number_of_subjects))
        for iteration in range(max_number_of_iterations):
            phi = norm.pdf(- 1.0 / std_old)
            Phi = norm.cdf(- 1.0 / std_old)
            std_new = 1.0 / math.sqrt(number_of_subjects * (1 - (phi / std_old) / (1 - Phi)) / ss)
            difference = math.fabs(std_new - std_old)
            if difference < convergence_tolerance:
                break
            else:
                std_old = std_new
            if iteration == max_number_of_iterations:
                msg = 'When initializing the acceleration std parameter from the empirical std, the fixed-point ' \
                      'algorithm did not satisfy the tolerance threshold within the allowed ' \
                      + str(max_number_of_iterations) + 'iterations. Difference = ' \
                      + str(difference) + ' > tolerance = ' + str(convergence_tolerance)
                warnings.warn(msg)
        return std_new

    # Standard deviations.
    heuristic_initial_time_shift_std = np.std(heuristic_initial_onset_ages)
    heuristic_initial_acceleration_std = get_acceleration_std_from_accelerations(heuristic_initial_accelerations)

    # Rescaling the initial momenta according to the mean of the acceleration factors.
    expected_mean_acceleration = float(truncnorm.stats(- 1.0 / heuristic_initial_acceleration_std, float('inf'),
                                                       loc=1.0, scale=heuristic_initial_acceleration_std, moments='m'))
    mean_acceleration = np.mean(heuristic_initial_accelerations)
    heuristic_initial_accelerations *= expected_mean_acceleration / mean_acceleration
    global_initial_momenta *= mean_acceleration / expected_mean_acceleration

    # Acceleration standard deviation, after whitening.
    heuristic_initial_acceleration_std = get_acceleration_std_from_accelerations(heuristic_initial_accelerations)

    logger.info('>> Estimated fixed effects:')
    logger.info('\t\t time_shift_std    =\t%.3f' % heuristic_initial_time_shift_std)
    logger.info('\t\t acceleration_std  =\t%.3f' % heuristic_initial_acceleration_std)

    logger.info('>> Estimated random effect statistics:')
    logger.info('\t\t onset_ages    =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
                (np.mean(heuristic_initial_onset_ages), heuristic_initial_time_shift_std))
    logger.info('\t\t accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
                (np.mean(heuristic_initial_accelerations), np.std(heuristic_initial_accelerations)))

    # Export the results -----------------------------------------------------------------------------------------------
    # Initial momenta.
    global_initial_momenta_path = os.path.join(global_path_to_data, 'ForInitialization__Momenta__RescaledWithHeuristics.txt')
    np.savetxt(global_initial_momenta_path, global_initial_momenta)

    # Onset ages.
    heuristic_initial_onset_ages_path = os.path.join(
        global_path_to_data, 'ForInitialization__OnsetAges__FromHeuristic.txt')
    np.savetxt(heuristic_initial_onset_ages_path, heuristic_initial_onset_ages)

    # Accelerations.
    heuristic_initial_accelerations_path = os.path.join(
        global_path_to_data, 'ForInitialization__Accelerations__FromHeuristic.txt')
    np.savetxt(heuristic_initial_accelerations_path, heuristic_initial_accelerations)

    # Modify the original model.xml file accordingly.
    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-momenta', global_initial_momenta_path)
    if heuristic_initial_time_shift_std > 0.0:
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-time-shift-std', '%.4f' % heuristic_initial_time_shift_std)
    if heuristic_initial_acceleration_std > 0.0:
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-acceleration-std', '%.4f' % heuristic_initial_acceleration_std)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-onset-ages', heuristic_initial_onset_ages_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-accelerations', heuristic_initial_accelerations_path)
    model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
    doc = parseString((et.tostring(
        model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    4]. Shoot from the average baseline age to the global average.
    --------------------------------------------------------------
        New values are obtained for the template, control points, and (longitudinal) momenta.
        Skipped if initial control points and momenta were given.
    """

    if not global_initial_control_points_are_given and not global_initial_momenta_are_given:

        logger.info('')
        logger.info('[ shoot from the average baseline age to the global average ]')

        # Shoot --------------------------------------------------------------------------------------------------------
        # Create folder.
        shooting_output_path = os.path.join(preprocessings_folder, '3_shooting_from_baseline_to_average')
        if os.path.isdir(shooting_output_path): shutil.rmtree(shooting_output_path)
        os.mkdir(shooting_output_path)

        # Instantiate a geodesic.
        geodesic = Geodesic(dense_mode=global_dense_mode,
                            kernel=kernel_factory.factory(xml_parameters.deformation_kernel_type,
                                                          kernel_width=xml_parameters.deformation_kernel_width),
                            use_rk2_for_shoot=xml_parameters.use_rk2_for_shoot,
                            use_rk2_for_flow=xml_parameters.use_rk2_for_flow,
                            t0=global_tmin, concentration_of_time_points=xml_parameters.concentration_of_time_points)
        geodesic.set_tmin(global_tmin)
        geodesic.set_tmax(global_t0)

        # Set the template, control points and momenta and update.
        geodesic.set_template_points_t0(
            {key: Variable(torch.from_numpy(value).type(global_tensor_scalar_type), requires_grad=False)
             for key, value in global_initial_template.get_points().items()})
        if global_dense_mode:
            geodesic.set_control_points_t0(geodesic.get_template_points_t0()['landmark_points'])
        else:
            geodesic.set_control_points_t0(Variable(torch.from_numpy(
                global_initial_control_points).type(global_tensor_scalar_type)))
        geodesic.set_momenta_t0(Variable(torch.from_numpy(
            global_initial_momenta).type(global_tensor_scalar_type), requires_grad=False))
        geodesic.update()

        # Adapt the global settings, for the custom output directory.
        global_deformetrica.output_dir = shooting_output_path

        # Write.
        geodesic.write('Shooting', global_objects_name, global_objects_name_extension, global_initial_template,
                       {key: Variable(torch.from_numpy(value).type(global_tensor_scalar_type), requires_grad=False)
                        for key, value in global_initial_template_data.items()}, global_deformetrica.output_dir,
                       write_adjoint_parameters=True)

        # Export results -----------------------------------------------------------------------------------------------
        number_of_timepoints = \
            geodesic.forward_exponential.number_of_time_points + geodesic.backward_exponential.number_of_time_points - 2

        # Template.
        for k, (object_name, object_name_extension) in enumerate(
                zip(global_objects_name, global_objects_name_extension)):
            # Copy the estimated template to the data folder.
            shooted_template_path = os.path.join(
                shooting_output_path, 'Shooting__GeodesicFlow__' + object_name + '__tp_' + str(number_of_timepoints)
                                      + ('__age_%.2f' % global_t0) + object_name_extension)
            global_initial_objects_template_path[k] = os.path.join(
                global_path_to_data, 'ForInitialization__Template_' + object_name + '__FromAtlasAndShooting' + object_name_extension)
            shutil.copyfile(shooted_template_path, global_initial_objects_template_path[k])

            if global_initial_objects_template_type[k] == 'PolyLine'.lower():
                cmd = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
                os.system(cmd)  # Quite time-consuming.
                if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                    os.remove(global_initial_objects_template_path[k] + '--')

        # Control points.
        shooted_control_points_path = os.path.join(
            shooting_output_path, 'Shooting__GeodesicFlow__ControlPoints__tp_' + str(number_of_timepoints)
                                  + ('__age_%.2f' % global_t0) + '.txt')
        global_initial_control_points_path = os.path.join(global_path_to_data,
                                                          'ForInitialization__ControlPoints__FromAtlasAndShooting.txt')
        shutil.copyfile(shooted_control_points_path, global_initial_control_points_path)

        # Momenta.
        shooted_momenta_path = os.path.join(
            shooting_output_path, 'Shooting__GeodesicFlow__Momenta__tp_' + str(number_of_timepoints)
                                  + ('__age_%.2f' % global_t0) + '.txt')
        global_initial_momenta_path = os.path.join(global_path_to_data, 'ForInitialization__Momenta__FromRegressionsAndShooting.txt')
        shutil.copyfile(shooted_momenta_path, global_initial_momenta_path)
        global_initial_momenta = read_3D_array(global_initial_momenta_path)

        # Modify and write the model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_template_spec_entry(model_xml_level0,
                                                                'filename', global_initial_objects_template_path)
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-control-points', global_initial_control_points_path)
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-momenta', global_initial_momenta_path)

        model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    5]. Tangent-space ICA on the individual momenta outputted by the atlas estimation.
    ----------------------------------------------------------------------------------
        Those momenta are first projected on the space orthogonal to the initial (longitudinal) momenta.
        Skipped if initial control points and modulation matrix were specified.
    """

    # Check if an initial (longitudinal) momenta is available.
    if not (global_initial_control_points_are_given and global_initial_modulation_matrix_is_given):

        logger.info('')
        logger.info('[ tangent-space ICA on the projected individual momenta ]')
        logger.info('')

        # Warning.
        if not global_initial_control_points_are_given and global_initial_modulation_matrix_is_given:
            msg = 'Initial modulation matrix is given but not the corresponding initial control points. ' \
                  'This given initial modulation matrix will be ignored, and overridden by a ICA-based heuristic.'
            warnings.warn(msg)

        # Read the current model xml parameters.
        xml_parameters = XmlParameters()
        xml_parameters._read_model_xml(model_xml_path)
        xml_parameters._read_dataset_xml(dataset_xml_path)
        xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

        # Load.
        control_points = read_2D_array(xml_parameters.initial_control_points)
        momenta = read_3D_array(os.path.join(atlas_output_path, atlas_type + 'Atlas__EstimatedParameters__Momenta.txt'))

        # Compute RKHS matrix.
        number_of_control_points = control_points.shape[0]
        dimension = control_points.shape[1]
        K = np.zeros((number_of_control_points * dimension, number_of_control_points * dimension))
        for i in range(number_of_control_points):
            for j in range(number_of_control_points):
                cp_i = control_points[i, :]
                cp_j = control_points[j, :]
                kernel_distance = math.exp(
                    - np.sum((cp_j - cp_i) ** 2) / (xml_parameters.deformation_kernel_width ** 2))
                for d in range(dimension):
                    K[dimension * i + d, dimension * j + d] = kernel_distance
                    K[dimension * j + d, dimension * i + d] = kernel_distance

        # Project.
        kernel = kernel_factory.factory('torch', kernel_width=xml_parameters.deformation_kernel_width)

        Km = np.dot(K, global_initial_momenta.ravel())
        mKm = np.dot(global_initial_momenta.ravel().transpose(), Km)

        w = []
        for i in range(momenta.shape[0]):
            w.append(momenta[i].ravel() - np.dot(momenta[i].ravel(), Km) / mKm * global_initial_momenta.ravel())
        w = np.array(w)

        # Dimensionality reduction.
        if xml_parameters.number_of_sources is not None:
            number_of_sources = xml_parameters.number_of_sources
        elif xml_parameters.initial_modulation_matrix is not None:
            number_of_sources = read_2D_array(xml_parameters.initial_modulation_matrix).shape[1]
        else:
            number_of_sources = 4
            logger.info('>> No initial modulation matrix given, neither a number of sources. '
                        'The latter will be ARBITRARILY defaulted to 4.')

        ica = FastICA(n_components=number_of_sources, max_iter=50000)
        global_initial_sources = ica.fit_transform(w)
        global_initial_modulation_matrix = ica.mixing_

        # Rescale.
        for s in range(number_of_sources):
            std = np.std(global_initial_sources[:, s])
            global_initial_sources[:, s] /= std
            global_initial_modulation_matrix[:, s] *= std

        # Print.
        residuals = []
        for i in range(global_number_of_subjects):
            residuals.append(w[i] - np.dot(global_initial_modulation_matrix, global_initial_sources[i]))
        mean_relative_residual = np.mean(np.absolute(np.array(residuals))) / np.mean(np.absolute(w))
        logger.info('>> Mean relative residual: %.3f %%.' % (100 * mean_relative_residual))

        # Save.
        global_initial_modulation_matrix_path = \
            os.path.join(global_path_to_data, 'ForInitialization__ModulationMatrix__FromICA.txt')
        np.savetxt(global_initial_modulation_matrix_path, global_initial_modulation_matrix)

        global_initial_sources_path = os.path.join(global_path_to_data, 'ForInitialization__Sources__FromICA.txt')
        np.savetxt(global_initial_sources_path, global_initial_sources)

        # Modify the original model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-modulation-matrix', global_initial_modulation_matrix_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-sources', global_initial_sources_path)
        model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

        logger.info('>> Estimated random effect statistics:')
        logger.info('\t\t sources =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
                    (np.mean(global_initial_sources), np.std(global_initial_sources)))

    """
    6]. Longitudinal registration of all target subjects.
    -----------------------------------------------------
        The reference is the average of the ages at all visits.
        The template, control points and modulation matrix are from the atlas estimation.
        The momenta is from the individual regressions.
    """

    logger.info('')
    logger.info('[ longitudinal registration of all subjects ]')
    logger.info('')

    # Clean folder.
    registration_output_path = os.path.join(preprocessings_folder, '4_longitudinal_registration')
    if os.path.isdir(registration_output_path):
        if global_overwrite:
            shutil.rmtree(registration_output_path)
        elif not os.path.isdir(os.path.join(registration_output_path, 'tmp')):
            registrations = os.listdir(registration_output_path)
            if len(registrations) > 0:
                shutil.rmtree(os.path.join(registration_output_path, os.listdir(registration_output_path)[-1]))
    if not os.path.isdir(registration_output_path): os.mkdir(registration_output_path)

    # Read the current longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Adapt the xml parameters and update.
    xml_parameters.model_type = 'LongitudinalRegistration'.lower()
    # xml_parameters.optimization_method_type = 'ScipyPowell'.lower()
    xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
    xml_parameters.convergence_tolerance = 1e-3
    xml_parameters.print_every_n_iters = 1

    # Adapt the global settings, for the custom output directory.
    global_deformetrica.output_dir = registration_output_path
    # xml_parameters._further_initialization(global_deformetrica.output_dir)

    # Launch.
    estimate_longitudinal_registration(global_deformetrica, xml_parameters, overwrite=global_overwrite)

    # Load results.
    estimated_onset_ages_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__OnsetAges.txt')
    estimated_accelerations_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__Accelerations.txt')
    estimated_sources_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__Sources.txt')

    global_onset_ages = read_2D_array(estimated_onset_ages_path)
    global_accelerations = read_2D_array(estimated_accelerations_path)
    global_sources = read_2D_array(estimated_sources_path)

    # Standard deviations.
    global_time_shift_std = np.std(global_onset_ages)
    global_acceleration_std = get_acceleration_std_from_accelerations(global_accelerations)

    # Rescaling the initial momenta according to the mean of the acceleration factors.
    expected_mean_acceleration = float(truncnorm.stats(- 1.0 / global_acceleration_std, float('inf'),
                                                       loc=1.0, scale=global_acceleration_std, moments='m'))
    mean_acceleration = np.mean(global_accelerations)
    global_accelerations *= expected_mean_acceleration / mean_acceleration
    global_initial_momenta *= mean_acceleration / expected_mean_acceleration

    # Acceleration standard deviation, after whitening.
    global_acceleration_std = get_acceleration_std_from_accelerations(global_accelerations)

    logger.info('')
    logger.info('>> Estimated fixed effects:')
    logger.info('\t\t time_shift_std    =\t%.3f' % global_time_shift_std)
    logger.info('\t\t acceleration_std  =\t%.3f' % global_acceleration_std)

    logger.info('>> Estimated random effect statistics:')
    logger.info('\t\t onset_ages    =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
                (np.mean(global_onset_ages), global_time_shift_std))
    logger.info('\t\t accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
                (np.mean(global_accelerations), np.std(global_accelerations)))
    logger.info('\t\t sources       =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
                (np.mean(global_sources), np.std(global_sources)))

    # Copy the output individual effects into the data folder.
    # Initial momenta.
    global_initial_momenta_path = os.path.join(
        global_path_to_data, 'ForInitialization__Momenta__RescaledWithLongitudinalRegistration.txt')
    np.savetxt(global_initial_momenta_path, global_initial_momenta)

    # Onset ages.
    global_initial_onset_ages_path = os.path.join(
        global_path_to_data, 'ForInitialization__OnsetAges__FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_onset_ages_path, global_initial_onset_ages_path)

    # Accelerations.
    global_initial_accelerations_path = os.path.join(
        global_path_to_data, 'ForInitialization__Accelerations__FromLongitudinalRegistration.txt')
    np.savetxt(global_initial_accelerations_path, global_accelerations)

    # Sources.
    global_initial_sources_path = os.path.join(
        global_path_to_data, 'ForInitialization__Sources__FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_sources_path, global_initial_sources_path)

    # Modify the original model.xml file accordingly.
    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-momenta', global_initial_momenta_path)
    if global_time_shift_std > 0:
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-time-shift-std', '%.4f' % global_time_shift_std)
    if global_acceleration_std > 0:
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-acceleration-std', '%.4f' % global_acceleration_std)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-onset-ages', global_initial_onset_ages_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-accelerations', global_initial_accelerations_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-sources', global_initial_sources_path)
    model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
    doc = parseString((et.tostring(
        model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    7]. Gradient-based optimization on population parameters.
    ---------------------------------------------------------
        Ignored if the user-specified optimization method is not the MCMC-SAEM.
    """

    logger.info('')
    logger.info('[ longitudinal atlas estimation with the GradientAscent optimizer ]')
    logger.info('')

    # Prepare and launch the longitudinal atlas estimation ---------------------------------------------------------
    # Clean folder.
    longitudinal_atlas_output_path = os.path.join(
        preprocessings_folder, '5_longitudinal_atlas_with_gradient_ascent')
    if os.path.isdir(longitudinal_atlas_output_path): shutil.rmtree(longitudinal_atlas_output_path)
    os.mkdir(longitudinal_atlas_output_path)

    # Read the current longitudinal model xml parameters, adapt them and update.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)
    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    xml_parameters.optimized_log_likelihood = 'class2'.lower()
    xml_parameters.max_line_search_iterations = 20
    xml_parameters.print_every_n_iters = 1

    # Adapt the global settings, for the custom output directory.
    global_deformetrica.output_dir = longitudinal_atlas_output_path
    # Settings().state_file = os.path.join(longitudinal_atlas_output_path, 'pydef_state.p')
    # xml_parameters._further_initialization(global_deformetrica.output_dir)

    # Launch.
    model = estimate_longitudinal_atlas(global_deformetrica, xml_parameters)

    # Export the results -------------------------------------------------------------------------------------------
    model_xml_level0 = et.parse(model_xml_path).getroot()

    # Template.
    for k, (object_name, object_name_extension) in enumerate(zip(global_objects_name,
                                                                 global_objects_name_extension)):
        estimated_template_path = os.path.join(
            longitudinal_atlas_output_path,
            'LongitudinalAtlas__EstimatedParameters__Template_%s__tp_%d__age_%.2f%s' %
            (object_name,
             model.spatiotemporal_reference_frame.geodesic.backward_exponential.number_of_time_points - 1,
             model.get_reference_time(), object_name_extension))
        global_initial_objects_template_path[k] = os.path.join(
            global_path_to_data,
            'ForInitialization__Template_%s__FromLongitudinalAtlas%s' % (object_name, object_name_extension))
        shutil.copyfile(estimated_template_path, global_initial_objects_template_path[k])

        if global_initial_objects_template_type[k] == 'PolyLine'.lower():
            cmd = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
            os.system(cmd)  # Quite time-consuming.
            if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                os.remove(global_initial_objects_template_path[k] + '--')

    model_xml_level0 = insert_model_xml_template_spec_entry(
        model_xml_level0, 'filename', global_initial_objects_template_path)

    # Control points.
    estimated_control_points_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ControlPoints.txt')
    global_initial_control_points_path = os.path.join(
        global_path_to_data, 'ForInitialization__ControlPoints__FromLongitudinalAtlas.txt')
    shutil.copyfile(estimated_control_points_path, global_initial_control_points_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-control-points', global_initial_control_points_path)

    # Momenta.
    estimated_momenta_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__Momenta.txt')
    global_initial_momenta_path = os.path.join(
        global_path_to_data, 'ForInitialization__Momenta__FromLongitudinalAtlas.txt')
    shutil.copyfile(estimated_momenta_path, global_initial_momenta_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-momenta', global_initial_momenta_path)

    # Modulation matrix.
    estimated_modulation_matrix_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ModulationMatrix.txt')
    global_initial_modulation_matrix_path = os.path.join(
        global_path_to_data, 'ForInitialization__ModulationMatrix__FromLongitudinalAtlas.txt')
    shutil.copyfile(estimated_modulation_matrix_path, global_initial_modulation_matrix_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-modulation-matrix', global_initial_modulation_matrix_path)

    # Reference time.
    estimated_reference_time_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ReferenceTime.txt')
    global_initial_reference_time = np.loadtxt(estimated_reference_time_path)
    model_xml_level0 = insert_model_xml_deformation_parameters_entry(
        model_xml_level0, 't0', '%.4f' % global_initial_reference_time)

    # Time-shift variance.
    estimated_time_shift_std_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__TimeShiftStd.txt')
    global_initial_time_shift_std = np.loadtxt(estimated_time_shift_std_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-time-shift-std', '%.4f' % global_initial_time_shift_std)

    # Acceleration variance.
    estimated_acceleration_std_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__AccelerationStd.txt')
    global_initial_acceleration_std = np.loadtxt(estimated_acceleration_std_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-acceleration-std', '%.4f' % global_initial_acceleration_std)

    # Noise variance.
    global_initial_noise_variance = model.get_noise_variance()
    global_initial_noise_std_string = ['{:.4f}'.format(math.sqrt(elt)) for elt in global_initial_noise_variance]
    model_xml_level0 = insert_model_xml_template_spec_entry(
        model_xml_level0, 'noise-std', global_initial_noise_std_string)

    # Onset ages.
    estimated_onset_ages_path = os.path.join(longitudinal_atlas_output_path,
                                             'LongitudinalAtlas__EstimatedParameters__OnsetAges.txt')
    global_initial_onset_ages_path = os.path.join(global_path_to_data, 'ForInitialization__OnsetAges__FromLongitudinalAtlas.txt')
    shutil.copyfile(estimated_onset_ages_path, global_initial_onset_ages_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-onset-ages', global_initial_onset_ages_path)

    # Accelerations.
    estimated_accelerations_path = os.path.join(
        longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__Accelerations.txt')
    global_initial_accelerations_path = os.path.join(
        global_path_to_data, 'ForInitialization__Accelerations__FromLongitudinalAtlas.txt')
    shutil.copyfile(estimated_accelerations_path, global_initial_accelerations_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-accelerations', global_initial_accelerations_path)

    # Sources.
    estimated_sources_path = os.path.join(longitudinal_atlas_output_path,
                                          'LongitudinalAtlas__EstimatedParameters__Sources.txt')
    global_initial_sources_path = os.path.join(global_path_to_data, 'ForInitialization__Sources__FromLongitudinalAtlas.txt')
    shutil.copyfile(estimated_sources_path, global_initial_sources_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-sources', global_initial_sources_path)

    # Finalization.
    model_xml_path = os.path.join(os.path.dirname(preprocessings_folder), 'initialized_model.xml')
    doc = parseString(
        (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')
