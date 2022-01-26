import os
import sys

sys.path.append('/home/benoit.sautydechalon/deformetrica')

import xml.etree.ElementTree as et

from deformetrica.in_out.xml_parameters import XmlParameters
from deformetrica.support.utilities.general_settings import Settings
from deformetrica import estimate_longitudinal_metric_model
from deformetrica.in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from sklearn import datasets, linear_model
from deformetrica.in_out.dataset_functions import read_and_create_scalar_dataset, read_and_create_image_dataset
from sklearn.decomposition import PCA
import deformetrica as dfca


def _initialize_modulation_matrix_and_sources(dataset, p0, v0, number_of_sources):
    unit_v0 = v0/np.linalg.norm(v0)
    unit_v0 = unit_v0.flatten()
    flat_p0 = p0.flatten()
    vectors = []
    for elt in dataset.deformable_objects:
        for e in elt: #To make it lighter in memory, and faster
            e_np = e.cpu().data.numpy()
            dimension = e_np.shape
            e_np = e_np.flatten()
            vector_projected = e_np - np.dot(e_np, unit_v0) * unit_v0
            vectors.append(vector_projected - flat_p0)

    logger.info("Performing principal component analysis on the orthogonal variations, for initialization of A and s_i.")

    # We now do a pca on those vectors
    pca = PCA(n_components=number_of_sources)
    pca.fit(vectors)
    if len(dimension) == 1:
        out = np.transpose(pca.components_)
    else:
        out = np.transpose(pca.components_).reshape((-1,) + dimension)
    for i in range(number_of_sources):
        out[:, i] /= np.linalg.norm(out[:, i])

    sources = []
    for elt in dataset.deformable_objects:
        obs_for_subject = np.array([(im.cpu().data.numpy() - p0).flatten() for im in elt])
        # We average the coordinate of these obs in pca space
        sources.append(np.mean(pca.transform(obs_for_subject), 0))

    return out, sources

def _smart_initialization_individual_effects(dataset):
    """
    least_square regression for each subject, so that yi = ai * t + bi
    output is the list of ais and bis
    this proceeds as if the initialization for the geodesic is a straight line
    """
    logger.info("Performing initial least square regressions on the subjects, for initialization purposes.")

    number_of_subjects = dataset.number_of_subjects
    dimension = dataset.deformable_objects[0][0].cpu().data.numpy().shape

    ais = []
    bis = []

    for i in range(number_of_subjects):

        # Special case of a single observation for the subject
        if len(dataset.times[i]) <= 1:
            ais.append(1.)
            bis.append(0.)

        else:

            least_squares = linear_model.LinearRegression()
            data_for_subject = np.array([elt.cpu().data.numpy().flatten() for elt in dataset.deformable_objects[i]])
            least_squares.fit(dataset.times[i].reshape(-1, 1), data_for_subject)

            a = least_squares.coef_.reshape(dimension)
            if type(a) == 'float' and a[0] < 0.001:
                a = np.array([0.001])
            ais.append(a)
            bis.append(least_squares.intercept_.reshape(dimension))

    return ais, bis

def _smart_initialization(dataset, number_of_sources, observation_type):

    observation_times = []
    for times in dataset.times:
        for t in times:
            observation_times.append(t)
    std_obs = np.std(observation_times)

    ais, bis = _smart_initialization_individual_effects(dataset)
    reference_time = np.mean([np.mean(times_i) for times_i in dataset.times])
    v0 = np.mean(ais, 0)

    p0 = 0
    for i in range(dataset.number_of_subjects):
        aux = np.mean(np.array([elt.cpu().data.numpy() for elt in dataset.deformable_objects[i]]), 0)
        p0 += aux
    p0 /= dataset.number_of_subjects


    alphas = []
    onset_ages = []
    for i in range(len(ais)):
        if Settings().dimension == 1:
            alpha_proposal = ais[i] / v0
            alpha = max(0.003, min(10., alpha_proposal))
            onset_age_proposal = 1. / alpha * (p0 - bis[i]) / v0
        else:
            alpha_proposal = np.dot(ais[i].flatten(), v0.flatten())/np.sum(v0**2)
            alpha = max(0.003, min(10., alpha_proposal))
            onset_age_proposal = 1. / alpha * np.dot(p0.flatten() - bis[i].flatten(), v0.flatten()) / np.sum(v0 ** 2)

        alphas.append(alpha)
        onset_age = max(reference_time - 2 * std_obs, min(reference_time + 2 * std_obs, onset_age_proposal))
        #logger.info(f"{onset_age_proposal}, {onset_age}")
        onset_ages.append(onset_age)


    # ADD a normalization step (0 mean, unit variance):
    if True:
        log_accelerations = np.log(alphas)
        log_accelerations = 0.5*(log_accelerations - np.mean(log_accelerations, 0))/np.std(log_accelerations, 0)
        alphas = np.exp(log_accelerations)
        # We want the onset ages to have an std equal to the std of the obser times


        onset_ages = (onset_ages - np.mean(onset_ages, 0))/np.std(onset_ages, 0) * std_obs + np.mean(onset_ages)
        #logger.info(f"std onset_ages vs obs times {np.std(onset_ages),std_obs}")

    reference_time = np.mean(onset_ages, 0)

    if number_of_sources > 0:
        modulation_matrix, sources = _initialize_modulation_matrix_and_sources(dataset, p0, v0, number_of_sources)

    else:
        modulation_matrix = None
        sources = None

    if True and sources is not None:
        sources = np.array(sources)
        sources = (sources - np.mean(sources, 0)) / np.std(sources, 0)

    return reference_time, v0, p0, np.array(onset_ages), np.array(alphas), modulation_matrix, sources


if __name__ == '__main__':

    logger.info('')
    logger.info('##############################')
    logger.info('##### PyDeformetrica 1.0 #####')
    logger.info('##############################')

    logger.info('')

    #TODO : remove this after debugging
    #assert len(sys.argv) == 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> "

    #model_xml_path = sys.argv[1]
    #dataset_xml_path = sys.argv[2]
    #optimization_parameters_xml_path = sys.argv[3]

    study = 'simulated_study/'

    model_xml_path = study + 'model.xml'
    dataset_xml_path = study + 'data_set.xml'
    optimization_parameters_xml_path = study + 'optimization_parameters_saem.xml'

    preprocessings_folder = 'preprocessing_metric_1'

    if not os.path.isdir(preprocessings_folder):
        os.mkdir(preprocessings_folder)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    deformetrica = dfca.Deformetrica(output_dir=preprocessings_folder, verbosity=logger.level)

    """
    1) Simple heuristic for initializing everything but the sources and the modulation matrix.
    """

    smart_initialization_output_path = os.path.join(preprocessings_folder, '1_smart_initialization')
    Settings().output_dir = smart_initialization_output_path
    Settings().dimension = xml_parameters.dimension

    if not os.path.isdir(smart_initialization_output_path):
        os.mkdir(smart_initialization_output_path)

    # Creating the dataset object
    dataset = read_and_create_scalar_dataset(xml_parameters)
    observation_type = 'scalar'

    # Heuristic for the initializations
    if xml_parameters.number_of_sources is None or xml_parameters.number_of_sources == 0:
        reference_time, average_a, p0, onset_ages, alphas, modulation_matrix, sources = _smart_initialization(dataset, 0, observation_type)
    else:
        reference_time, average_a, p0, onset_ages, alphas, modulation_matrix, sources = _smart_initialization(dataset, xml_parameters.number_of_sources, observation_type)

    # We save the onset ages and alphas.
    # We then set the right path in the xml_parameters, for the proper initialization.
    write_2D_array(np.log(alphas), smart_initialization_output_path, "SmartInitialization_log_accelerations.txt")
    xml_parameters.initial_accelerations = os.path.join(smart_initialization_output_path, "SmartInitialization_log_accelerations.txt")

    write_2D_array(onset_ages, smart_initialization_output_path, "SmartInitialization_onset_ages.txt")
    xml_parameters.initial_onset_ages = os.path.join(smart_initialization_output_path, "SmartInitialization_onset_ages.txt")

    if xml_parameters.exponential_type != 'deep':
        write_2D_array(np.array([p0]), smart_initialization_output_path, "SmartInitialization_p0.txt")
        xml_parameters.p0 = os.path.join(smart_initialization_output_path, "SmartInitialization_p0.txt")

        write_2D_array(np.array([average_a]), smart_initialization_output_path, "SmartInitialization_v0.txt")
        xml_parameters.v0 = os.path.join(smart_initialization_output_path, "SmartInitialization_v0.txt")

    if modulation_matrix is not None:
        assert sources is not None
        if xml_parameters.exponential_type != 'deep':
            write_2D_array(modulation_matrix,  smart_initialization_output_path, "SmartInitialization_modulation_matrix.txt")
            xml_parameters.initial_modulation_matrix = os.path.join(smart_initialization_output_path, "SmartInitialization_modulation_matrix.txt")
        write_2D_array(sources, smart_initialization_output_path, "SmartInitialization_sources.txt")
        xml_parameters.initial_sources = os.path.join(smart_initialization_output_path,
                                                          "SmartInitialization_sources.txt")

    xml_parameters.t0 = reference_time

    # Now the stds:
    xml_parameters.initial_acceleration_variance = np.var(np.log(alphas))
    xml_parameters.initial_time_shift_variance = np.var(onset_ages)

    """
    2) Gradient descent on the mode
    """

    mode_descent_output_path = os.path.join(preprocessings_folder, '2_gradient_descent_on_the_mode')
    # To perform this gradient descent, we use the iniialization heuristic, starting from
    # a flat metric and linear regressions one each subject

    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    xml_parameters.scale_initial_step_size = True
    xml_parameters.initialize = True
    xml_parameters.max_iterations = 33
    xml_parameters.initial_step_size = .01
    xml_parameters.max_line_search_iterations = 4
    xml_parameters.convergence_tolerance = 1e-5
    xml_parameters.save_every_n_iters = 1
    xml_parameters.print_every_n_iters = 1

    # Freezing some variances !
    xml_parameters.freeze_acceleration_variance = True
    xml_parameters.freeze_noise_variance = True
    xml_parameters.freeze_onset_age_variance = True

    # Freezing other variables
    xml_parameters.freeze_modulation_matrix = False
    xml_parameters.freeze_sources = False
    xml_parameters.freeze_p0 = False
    xml_parameters.freeze_v0 = False
    
    xml_parameters.output_dir = mode_descent_output_path
    Settings().output_dir = mode_descent_output_path

    logger.info(" >>> Performing gradient descent on the mode.")

    # First few iterations to get a descent metric
    estimate_longitudinal_metric_model(xml_parameters, logger=logger)


    """"""""""""""""""""""""""""""""
    """Creating a xml file"""
    """"""""""""""""""""""""""""""""

    model_xml = et.Element('data-set')
    model_xml.set('deformetrica-min-version', "3.0.0")

    model_type = et.SubElement(model_xml, 'model-type')
    model_type.text = "LongitudinalMetricLearning"

    dimension = et.SubElement(model_xml, 'dimension')
    dimension.text=str(Settings().dimension)

    estimated_alphas = np.loadtxt(os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_alphas.txt'))
    estimated_onset_ages = np.loadtxt(os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_onset_ages.txt'))

    initial_time_shift_std = et.SubElement(model_xml, 'initial-time-shift-std')
    initial_time_shift_std.text = str(np.std(estimated_onset_ages))

    initial_log_acceleration_std = et.SubElement(model_xml, 'initial-log-acceleration-std')
    initial_log_acceleration_std.text = str(np.std(np.log(estimated_alphas)))

    deformation_parameters = et.SubElement(model_xml, 'deformation-parameters')

    exponential_type = et.SubElement(deformation_parameters, 'exponential-type')
    exponential_type.text = xml_parameters.exponential_type

    if xml_parameters.exponential_type == 'parametric':
        interpolation_points = et.SubElement(deformation_parameters, 'interpolation-points-file')
        interpolation_points.text = os.path.join(study + mode_descent_output_path, 'LongitudinalMetricModel_interpolation_points.txt')
        kernel_width = et.SubElement(deformation_parameters, 'kernel-width')
        kernel_width.text = str(xml_parameters.deformation_kernel_width)

    concentration_of_timepoints = et.SubElement(deformation_parameters,
                                                'concentration-of-timepoints')
    concentration_of_timepoints.text = str(xml_parameters.concentration_of_time_points)

    number_of_timepoints = et.SubElement(deformation_parameters,
                                                'number-of-timepoints')
    number_of_timepoints.text = str(xml_parameters.number_of_time_points)

    estimated_fixed_effects = np.load(os.path.join(mode_descent_output_path,
                                                   'LongitudinalMetricModel_all_fixed_effects.npy'), allow_pickle=True)[()]

    if xml_parameters.exponential_type in ['parametric']: # otherwise it's not saved !
        metric_parameters_file = et.SubElement(deformation_parameters,'metric-parameters-file')
        metric_parameters_file.text = os.path.join(study + mode_descent_output_path, 'LongitudinalMetricModel_metric_parameters.txt')

    if xml_parameters.number_of_sources is not None and xml_parameters.number_of_sources > 0:
        initial_sources_file = et.SubElement(model_xml, 'initial-sources')
        initial_sources_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_sources.txt')
        number_of_sources = et.SubElement(deformation_parameters, 'number-of-sources')
        number_of_sources.text = str(xml_parameters.number_of_sources)
        initial_modulation_matrix_file = et.SubElement(model_xml, 'initial-modulation-matrix')
        initial_modulation_matrix_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_modulation_matrix.txt')

    t0 = et.SubElement(deformation_parameters, 't0')
    t0.text = str(estimated_fixed_effects['reference_time'])

    v0 = et.SubElement(deformation_parameters, 'v0')
    v0.text = os.path.join(study + mode_descent_output_path, 'LongitudinalMetricModel_v0.txt')

    p0 = et.SubElement(deformation_parameters, 'p0')
    p0.text = os.path.join(study + mode_descent_output_path, 'LongitudinalMetricModel_p0.txt')

    initial_onset_ages = et.SubElement(model_xml, 'initial-onset-ages')
    initial_onset_ages.text = os.path.join(mode_descent_output_path,
                                           "LongitudinalMetricModel_onset_ages.txt")

    initial_log_accelerations = et.SubElement(model_xml, 'initial-log-accelerations')
    initial_log_accelerations.text = os.path.join(mode_descent_output_path,
                                                  "LongitudinalMetricModel_log_accelerations.txt")

    model_xml_path = study + 'model_after_initialization_metric_1.xml'
    doc = parseString((et.tostring(model_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')
