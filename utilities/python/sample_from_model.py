import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import math
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from numpy.random import poisson, exponential, normal
import warnings

from deformetrica import get_model_options
from api.deformetrica import Deformetrica
from core.models.longitudinal_atlas import LongitudinalAtlas

from in_out.xml_parameters import XmlParameters
from core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from launch.estimate_longitudinal_metric_model import instantiate_longitudinal_metric_model
from in_out.deformable_object_reader import DeformableObjectReader
from in_out.dataset_functions import create_dataset
from in_out.array_readers_and_writers import *


def add_gaussian_noise_to_vtk_file(global_output_dir, filename, obj_type, noise_std):
    reader = DeformableObjectReader()
    obj = reader.create_object(filename, obj_type)
    obj.update()
    obj.set_points(obj.points + normal(0.0, noise_std, size=obj.points.shape))
    obj.write(global_output_dir, os.path.basename(filename))


if __name__ == '__main__':

    """
    Basic info printing.
    """

    logger.info('')
    logger.info('##############################')
    logger.info('##### PyDeformetrica 1.0 #####')
    logger.info('##############################')
    logger.info('')

    """
    Read command line, create output directory, read the model xml file.
    """

    assert len(sys.argv) in [4, 5, 6, 7], \
        'Usage: ' + sys.argv[0] + " <model.xml> { <number_of_subjects> " \
                                  "<mean_number_of_visits_minus_two> " "<mean_observation_time_window> } " \
                                  "OR { path_to_visit_ages_file.txt } <optional --add_noise>"

    model_xml_path = sys.argv[1]

    global_add_noise = False
    if len(sys.argv) in [5, 6]:
        number_of_subjects = int(sys.argv[2])
        mean_number_of_visits_minus_two = float(sys.argv[3])
        mean_observation_time_window = float(sys.argv[4])

        if len(sys.argv) == 6:
            if sys.argv[5] in ['--add_noise', '--add-noise']:
                global_add_noise = True
            else:
                msg = 'Unknown command-line option: "%s". Ignoring.' % sys.argv[5]
                warnings.warn(msg)

    elif len(sys.argv) in [3, 4]:
        path_to_visit_ages_file = sys.argv[2]
        visit_ages = read_2D_list(path_to_visit_ages_file)
        number_of_subjects = len(visit_ages)

        if len(sys.argv) == 4:
            if sys.argv[3] == '--add_noise':
                global_add_noise = True
            else:
                msg = 'Unknown command-line option: "%s". Ignoring.' % sys.argv[3]
                warnings.warn(msg)

    else:
        raise RuntimeError('Incorrect number of arguments.')

    sample_index = 1
    sample_folder = 'sample_' + str(sample_index)
    while os.path.isdir(sample_folder):
        sample_index += 1
        sample_folder = 'sample_' + str(sample_index)
    os.mkdir(sample_folder)
    global_output_dir = sample_folder

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)

    template_specifications = xml_parameters.template_specifications
    model_options = get_model_options(xml_parameters)
    model_options['tensor_scalar_type'] = torch.DoubleTensor
    model_options['tensor_integer_type'] = torch.LongTensor

    global_dimension = model_options['dimension']

    # deformetrica = Deformetrica()
    # (template_specifications, model_options, _) = deformetrica.further_initialization(
    #     xml_parameters.model_type, xml_parameters.template_specifications, get_model_options(xml_parameters))

    if xml_parameters.model_type == 'LongitudinalAtlas'.lower():

        """
        Instantiate the model.
        """
        model = LongitudinalAtlas(template_specifications, **model_options)
        if np.min(model.get_noise_variance()) < 0: model.set_noise_variance(np.array([0.0]))

        """
        Draw random visit ages and create a degenerated dataset object.
        """

        if len(sys.argv) in [5, 6]:
            visit_ages = []
            for i in range(number_of_subjects):
                number_of_visits = 2 + poisson(mean_number_of_visits_minus_two)
                observation_time_window = normal(mean_observation_time_window,
                                                 math.sqrt(model.get_time_shift_variance()))

                time_between_two_consecutive_visits = observation_time_window / float(number_of_visits - 1)
                age_at_baseline = normal(model.get_reference_time(), math.sqrt(model.get_time_shift_variance())) \
                                  - 0.5 * observation_time_window

                ages = [age_at_baseline + j * time_between_two_consecutive_visits for j in range(number_of_visits)]
                visit_ages.append(ages)

        subject_ids = ['s' + str(i) for i in range(number_of_subjects)]
        dataset = LongitudinalDataset(subject_ids, times=visit_ages)

        logger.info('>> %d subjects will be generated, with %.2f visits on average, covering an average period of %.2f years.'
              % (number_of_subjects, float(dataset.total_number_of_observations) / float(number_of_subjects),
                 np.mean(np.array([ages[-1] - ages[0] for ages in dataset.times]))))

        """
        Generate individual RER.
        """

        # Complementary xml parameters.
        t0 = xml_parameters.t0
        tmin = xml_parameters.tmin
        tmax = xml_parameters.tmax

        if tmin == float('inf'):
            tmin *= -1
        if tmax == - float('inf'):
            tmax *= -1

        sources_mean = 0.0
        sources_std = 1.0
        if xml_parameters.initial_sources_mean is not None:
            sources_mean = read_2D_array(xml_parameters.initial_sources_mean)
        if xml_parameters.initial_sources_std is not None:
            sources_std = read_2D_array(xml_parameters.initial_sources_std)

        onset_ages = np.zeros((number_of_subjects,))
        accelerations = np.zeros((number_of_subjects,))
        sources = np.zeros((number_of_subjects, model.number_of_sources)) + sources_mean

        i = 0
        while i in range(number_of_subjects):
            onset_ages[i] = model.individual_random_effects['onset_age'].sample()
            accelerations[i] = model.individual_random_effects['acceleration'].sample()
            sources[i] = model.individual_random_effects['sources'].sample() * sources_std

            min_age = accelerations[i] * (visit_ages[i][0] - onset_ages[i]) + t0
            max_age = accelerations[i] * (visit_ages[i][-1] - onset_ages[i]) + t0
            if min_age >= tmin and max_age <= tmax:
                i += 1

        # dataset.times = visit_ages

        individual_RER = {}
        individual_RER['sources'] = sources
        individual_RER['onset_age'] = onset_ages
        individual_RER['acceleration'] = accelerations

        """
        Call the write method of the model.
        """

        model.name = 'SimulatedData'
        model.write(dataset, None, individual_RER, global_output_dir, update_fixed_effects=False, write_residuals=False)

        if global_dimension == 2:
            cmd_replace = 'sed -i -- s/POLYGONS/LINES/g ' + global_output_dir + '/*Reconstruction*'
            cmd_delete = 'rm ' + global_output_dir + '/*--'
            cmd = cmd_replace + ' && ' + cmd_delete
            os.system(cmd)  # Quite time-consuming.

        """
        Optionally add gaussian noise to the generated samples.
        """

        if global_add_noise:
            assert np.min(model.get_noise_variance()) > 0, 'Invalid noise variance.'
            objects_type = [elt['deformable_object_type'] for elt in xml_parameters.template_specifications.values()]
            for i in range(number_of_subjects):
                for j, age in enumerate(dataset.times[i]):
                    for k, (obj_type, obj_name, obj_extension, obj_noise) in enumerate(zip(
                            objects_type, model.objects_name, model.objects_name_extension,
                            model.get_noise_variance())):
                        filename = 'sample_%d/SimulatedData__Reconstruction__%s__subject_s%d__tp_%d__age_%.2f%s' \
                                   % (sample_index, obj_name, i, j, age, obj_extension)
                        add_gaussian_noise_to_vtk_file(global_output_dir, filename, obj_type, math.sqrt(obj_noise))

            if global_dimension == 2:
                cmd_replace = 'sed -i -- s/POLYGONS/LINES/g ' + global_output_dir + '/*Reconstruction*'
                cmd_delete = 'rm ' + global_output_dir + '/*--'
                cmd = cmd_replace + ' && ' + cmd_delete
                os.system(cmd)  # Quite time-consuming.

        """
        Create and save the dataset xml file.
        """

        dataset_xml = et.Element('data-set')
        dataset_xml.set('deformetrica-min-version', "3.0.0")

        for i in range(number_of_subjects):

            subject_id = 'sub-' + str(i)
            subject_xml = et.SubElement(dataset_xml, 'subject')
            subject_xml.set('id', subject_id)

            for j, age in enumerate(dataset.times[i]):

                visit_id = 'ses-' + str(j)
                visit_xml = et.SubElement(subject_xml, 'visit')
                visit_xml.set('id', visit_id)
                age_xml = et.SubElement(visit_xml, 'age')
                age_xml.text = '%.2f' % age

                for k, (obj_name, obj_extension) in enumerate(zip(model.objects_name, model.objects_name_extension)):
                    filename_xml = et.SubElement(visit_xml, 'filename')
                    filename_xml.text = 'sample_%d/SimulatedData__Reconstruction__%s__subject_s%d__tp_%d__age_%.2f%s' \
                                        % (sample_index, obj_name, i, j, age, obj_extension)
                    filename_xml.set('object_id', obj_name)

        dataset_xml_path = 'data_set__sample_' + str(sample_index) + '.xml'
        doc = parseString((et.tostring(dataset_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(dataset_xml_path, [doc], fmt='%s')

        """
        Create a dataset object from the xml, and compute the residuals.
        """

        xml_parameters._read_dataset_xml(dataset_xml_path)
        dataset = create_dataset(xml_parameters.template_specifications,
                                 visit_ages=xml_parameters.visit_ages,
                                 dataset_filenames=xml_parameters.dataset_filenames,
                                 subject_ids=xml_parameters.subject_ids,
                                 dimension=global_dimension)

        if global_add_noise:
            (template_data, template_points, control_points,
             momenta, modulation_matrix) = model._fixed_effects_to_torch_tensors(False)
            sources, onset_ages, accelerations = model._individual_RER_to_torch_tensors(individual_RER, False)
            absolute_times, tmin, tmax = model._compute_absolute_times(dataset.times, onset_ages, accelerations)
            model._update_spatiotemporal_reference_frame(
                template_points, control_points, momenta, modulation_matrix, tmin, tmax)
            residuals = model._compute_residuals(dataset, template_data, absolute_times, sources)

            residuals_list = [[[residuals_i_j_k.detach().cpu().numpy() for residuals_i_j_k in residuals_i_j]
                               for residuals_i_j in residuals_i] for residuals_i in residuals]
            write_3D_list(residuals_list, global_output_dir, model.name + "__EstimatedParameters__Residuals.txt")

            # Print empirical noise if relevant.
            assert np.min(model.get_noise_variance()) > 0, 'Invalid noise variance.'
            objects_empirical_noise_std = np.zeros((len(residuals_list[0][0])))
            for i in range(len(residuals_list)):
                for j in range(len(residuals_list[i])):
                    for k in range(len(residuals_list[i][j])):
                        objects_empirical_noise_std[k] += residuals_list[i][j][k]
            for k in range(len(residuals_list[0][0])):
                objects_empirical_noise_std[k] = \
                    math.sqrt(objects_empirical_noise_std[k]
                              / float(dataset.total_number_of_observations * model.objects_noise_dimension[k]))
                logger.info('>> Empirical noise std for object "%s": %.4f'
                      % (model.objects_name[k], objects_empirical_noise_std[k]))
            write_2D_array(objects_empirical_noise_std,
                           global_output_dir, model.name + '__EstimatedParameters__EmpiricalNoiseStd.txt')

    elif xml_parameters.model_type == 'LongitudinalMetricLearning'.lower():

        """
        Instantiate the model.
        """
        model, _ = instantiate_longitudinal_metric_model(xml_parameters, dataset=None,
                                                         number_of_subjects=number_of_subjects,
                                                         observation_type='image')
        assert model.get_noise_variance() is not None \
               and model.get_noise_variance() > 0., "Please provide a noise variance"

        """
        Draw random visit ages and create a degenerated dataset object.
        """

        visit_ages = []
        for i in range(number_of_subjects):
            number_of_visits = 2 + poisson(mean_number_of_visits_minus_two)
            observation_time_window = exponential(mean_observation_time_window)

            time_between_two_consecutive_visits = observation_time_window / float(number_of_visits - 1)
            age_at_baseline = normal(model.get_reference_time(), math.sqrt(model.get_onset_age_variance())) \
                              - 0.5 * observation_time_window

            ages = [age_at_baseline + j * time_between_two_consecutive_visits for j in range(number_of_visits)]
            visit_ages.append(np.array(ages))

        dataset = LongitudinalDataset()
        dataset.times = visit_ages
        dataset.subject_ids = ['s' + str(i) for i in range(number_of_subjects)]
        dataset.number_of_subjects = number_of_subjects
        dataset.total_number_of_observations = sum([len(elt) for elt in visit_ages])

        logger.info('>> %d subjects will be generated, with %.2f visits on average, covering an average period of %.2f years.'
              % (number_of_subjects, float(dataset.total_number_of_observations) / float(number_of_subjects),
                 np.mean(np.array([ages[-1] - ages[0] for ages in dataset.times]))))

        """
        Generate metric parameters.
        """
        if xml_parameters.metric_parameters_file is None:
            logger.info("The generation of metric parameters is only handled in one dimension")
            values = np.random.binomial(1, 0.5, xml_parameters.number_of_interpolation_points)
            values = values / np.sum(values)
            model.set_metric_parameters(values)

        """
        Generate individual RER.
        """

        onset_ages = np.zeros((number_of_subjects,))
        log_accelerations = np.zeros((number_of_subjects,))
        sources = np.zeros((number_of_subjects, xml_parameters.number_of_sources))

        for i in range(number_of_subjects):
            onset_ages[i] = model.individual_random_effects['onset_age'].sample()
            log_accelerations[i] = model.individual_random_effects['log_acceleration'].sample()
            sources[i] = model.individual_random_effects['sources'].sample()

        individual_RER = {}
        individual_RER['onset_age'] = onset_ages
        individual_RER['log_acceleration'] = log_accelerations
        individual_RER['sources'] = sources

        """
        Call the write method of the model.
        """

        model.name = 'SimulatedData'
        model.write(dataset, None, individual_RER, sample=True)

        # Create a dataset xml for the simulations which will
        dataset_xml = et.Element('data-set')
        dataset_xml.set('deformetrica-min-version', "3.0.0")

        if False:
            group_file = et.SubElement(dataset_xml, 'group-file')
            group_file.text = "sample_%d/SimulatedData_subject_ids.txt" % (sample_index)

            observations_file = et.SubElement(dataset_xml, 'observations-file')
            observations_file.text = "sample_%d/SimulatedData_generated_values.txt" % (sample_index)

            timepoints_file = et.SubElement(dataset_xml, 'timepoints-file')
            timepoints_file.text = "sample_%d/SimulatedData_times.txt" % (sample_index)

            dataset_xml_path = 'data_set__sample_' + str(sample_index) + '.xml'
            doc = parseString(
                (et.tostring(dataset_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
            np.savetxt(dataset_xml_path, [doc], fmt='%s')

        else:  # Image dataset
            dataset_xml = et.Element('data-set')
            dataset_xml.set('deformetrica-min-version', "3.0.0")

            for i in range(number_of_subjects):

                subject_id = 's' + str(i)
                subject_xml = et.SubElement(dataset_xml, 'subject')
                subject_xml.set('id', subject_id)

                for j, age in enumerate(dataset.times[i]):
                    visit_id = 'ses-' + str(j)
                    visit_xml = et.SubElement(subject_xml, 'visit')
                    visit_xml.set('id', visit_id)
                    age_xml = et.SubElement(visit_xml, 'age')
                    age_xml.text = str(age)

                    filename_xml = et.SubElement(visit_xml, 'filename')
                    filename_xml.set('object_id', 'starfish')
                    filename_xml.text = os.path.join(global_output_dir, 'subject_'+str(i),
                                                 model.name + "_" + str(dataset.subject_ids[i])+ "_t__" + str(age) + ".npy")


            dataset_xml_path = 'data_set__sample_' + str(sample_index) + '.xml'
            doc = parseString(
                (et.tostring(dataset_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
            np.savetxt(dataset_xml_path, [doc], fmt='%s')

    else:
        msg = 'Sampling from the specified "' + xml_parameters.model_type + '" model is not available yet.'
        raise RuntimeError(msg)
