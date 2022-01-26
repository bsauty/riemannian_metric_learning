import math
import torch

from in_out.array_readers_and_writers import *
from in_out.image_functions import points_to_voxels_transform, metric_to_image_radial_length


import logging
logger = logging.getLogger(__name__)


def initialize_control_points(initial_control_points, template, spacing, deformation_kernel_width,
                              dimension, dense_mode):
    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
        logger.info('>> Reading %d initial control points from file %s.' % (len(control_points), initial_control_points))

    else:
        if not dense_mode:
            control_points = create_regular_grid_of_points(template.bounding_box, spacing, dimension)
            # if len(template.object_list) == 1 and template.object_list[0].type.lower() == 'image':
            #     control_points = remove_useless_control_points(control_points, template.object_list[0],
            #                                                    deformation_kernel_width)
            logger.info('>> Set of %d control points defined.' % len(control_points))
        else:
            assert (('landmark_points' in template.get_points().keys()) and
                    ('image_points' not in template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template.get_points()['landmark_points']

    return control_points


def initialize_momenta(initial_momenta, number_of_control_points, dimension, number_of_subjects=0, random=False):
    if initial_momenta is not None:
        momenta = read_3D_array(initial_momenta)
        logger.info('>> Reading initial momenta from file: %s.' % initial_momenta)

    else:
        if number_of_subjects == 0:
            if random:
                momenta = np.random.randn(number_of_control_points, dimension) / math.sqrt(number_of_control_points * dimension)
                logger.info('>> Momenta randomly initialized.')
            else:
                momenta = np.zeros((number_of_control_points, dimension))
                logger.info('>> Momenta initialized to zero.')
        else:
            momenta = np.zeros((number_of_subjects, number_of_control_points, dimension))
            logger.info('>> Momenta initialized to zero, for %d subjects.' % number_of_subjects)

    return momenta


def initialize_covariance_momenta_inverse(control_points, kernel, dimension):
    return np.kron(kernel.get_kernel_matrix(torch.from_numpy(control_points)).detach().numpy(), np.eye(dimension))


def initialize_modulation_matrix(initial_modulation_matrix, number_of_control_points, number_of_sources, dimension):
    if initial_modulation_matrix is not None:
        modulation_matrix = read_2D_array(initial_modulation_matrix)
        if len(modulation_matrix.shape) == 1:
            modulation_matrix = modulation_matrix.reshape(-1, 1)
        logger.info('>> Reading ' + str(
            modulation_matrix.shape[1]) + '-source initial modulation matrix from file: ' + initial_modulation_matrix)

    else:
        if number_of_sources is None:
            raise RuntimeError(
                'The number of sources must be set before calling the update method of the LongitudinalAtlas class.')
        modulation_matrix = np.zeros((number_of_control_points * dimension, number_of_sources))

    return modulation_matrix


def initialize_sources(initial_sources, number_of_subjects, number_of_sources):
    if initial_sources is not None:
        sources = read_2D_array(initial_sources).reshape((-1, number_of_sources))
        logger.info('>> Reading initial sources from file: ' + initial_sources)
    else:
        sources = np.zeros((number_of_subjects, number_of_sources))
        logger.info('>> Initializing all sources to zero')
    return sources


def initialize_onset_ages(initial_onset_ages, number_of_subjects, reference_time):
    if initial_onset_ages is not None:
        onset_ages = read_2D_array(initial_onset_ages)
        logger.info('>> Reading initial onset ages from file: ' + initial_onset_ages)
    else:
        onset_ages = np.zeros((number_of_subjects,)) + reference_time
        logger.info('>> Initializing all onset ages to the initial reference time: %.2f' % reference_time)
    return onset_ages


def initialize_accelerations(initial_accelerations, number_of_subjects):
    if initial_accelerations is not None:
        accelerations = read_2D_array(initial_accelerations)
        logger.info('>> Reading initial accelerations from file: ' + initial_accelerations)
    else:
        accelerations = np.ones((number_of_subjects,))
        logger.info('>> Initializing all accelerations to one.')
    return accelerations


def create_regular_grid_of_points(box, spacing, dimension):
    """
    Creates a regular grid of 2D or 3D points, as a numpy array of size nb_of_points x dimension.
    box: (dimension, 2)
    """

    axis = []
    for d in range(dimension):
        min = box[d, 0]
        max = box[d, 1]
        length = max - min
        assert (length >= 0.0)

        offset = 0.5 * (length - spacing * math.floor(length / spacing))
        axis.append(np.arange(min + offset, max + 1e-10, spacing))

    if dimension == 1:
        control_points = np.zeros((len(axis[0]), dimension))
        control_points[:, 0] = axis[0].flatten()

    elif dimension == 2:
        x_axis, y_axis = np.meshgrid(axis[0], axis[1])

        assert (x_axis.shape == y_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()

    elif dimension == 3:
        x_axis, y_axis, z_axis = np.meshgrid(axis[0], axis[1], axis[2])

        assert (x_axis.shape == y_axis.shape)
        assert (x_axis.shape == z_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()
        control_points[:, 2] = z_axis.flatten()

    elif dimension == 4:
        x_axis, y_axis, z_axis, t_axis = np.meshgrid(axis[0], axis[1], axis[2], axis[3])

        assert (x_axis.shape == y_axis.shape)
        assert (x_axis.shape == z_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()
        control_points[:, 2] = z_axis.flatten()
        control_points[:, 3] = t_axis.flatten()

    else:
        raise RuntimeError('Invalid ambient space dimension.')

    return control_points


def remove_useless_control_points(control_points, image, kernel_width):
    control_voxels = points_to_voxels_transform(control_points, image.affine)  # To be modified if image + mesh case.
    kernel_voxel_width = metric_to_image_radial_length(kernel_width, image.affine)

    intensities = image.get_intensities()
    image_shape = intensities.shape

    threshold = 1e-5
    region_size = 2 * kernel_voxel_width

    final_control_points = []
    for control_point, control_voxel in zip(control_points, control_voxels):

        axes = []
        for d in range(image.dimension):
            axe = np.arange(max(int(control_voxel[d] - region_size), 0),
                            min(int(control_voxel[d] + region_size), image_shape[d] - 1))
            axes.append(axe)

        neighbouring_voxels = np.array(np.meshgrid(*axes))
        for d in range(image.dimension):
            neighbouring_voxels = np.swapaxes(neighbouring_voxels, d, d + 1)
        neighbouring_voxels = neighbouring_voxels.reshape(-1, image.dimension)

        if (image.dimension == 2 and np.any(intensities[neighbouring_voxels[:, 0],
                                                        neighbouring_voxels[:, 1]] > threshold)) \
                or (image.dimension == 3 and np.any(intensities[neighbouring_voxels[:, 0],
                                                                neighbouring_voxels[:, 1],
                                                                neighbouring_voxels[:, 2]] > threshold)):
            final_control_points.append(control_point)

    return np.array(final_control_points)
