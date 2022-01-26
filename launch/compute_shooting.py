import torch

from core import default
from core.model_tools.deformations.geodesic import Geodesic
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata
from support import kernels as kernel_factory

import logging
logger = logging.getLogger(__name__)


def compute_shooting(template_specifications,
                     dimension=default.dimension,
                     tensor_scalar_type=default.tensor_scalar_type,
                     tensor_integer_type=default.tensor_integer_type,

                     deformation_kernel_type=default.deformation_kernel_type,
                     deformation_kernel_width=default.deformation_kernel_width,
                     deformation_kernel_device=default.deformation_kernel_device,

                     shoot_kernel_type=None,
                     initial_control_points=default.initial_control_points,
                     initial_momenta=default.initial_momenta,
                     concentration_of_time_points=default.concentration_of_time_points,
                     t0=None, tmin=default.tmin, tmax=default.tmax, dense_mode=default.dense_mode,
                     number_of_time_points=default.number_of_time_points,
                     use_rk2_for_shoot=default.use_rk2_for_shoot,
                     use_rk2_for_flow=default.use_rk2_for_flow,
                     gpu_mode=default.gpu_mode,
                     output_dir=default.output_dir, **kwargs
                     ):
    logger.info('[ compute_shooting function ]')

    """
    Create the template object
    """

    deformation_kernel = kernel_factory.factory(deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=deformation_kernel_width)

    (object_list, t_name, t_name_extension,
     t_noise_variance, multi_object_attachment) = create_template_metadata(
        template_specifications, dimension, gpu_mode=gpu_mode)

    template = DeformableMultiObject(object_list)

    """
    Reading Control points and momenta
    """

    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
    else:
        raise RuntimeError('Please specify a path to control points to perform a shooting')

    if initial_momenta is not None:
        momenta = read_3D_array(initial_momenta)
    else:
        raise RuntimeError('Please specify a path to momenta to perform a shooting')

    # _, b = control_points.shape
    # assert Settings().dimension == b, 'Please set the correct dimension in the model.xml file.'

    momenta_torch = torch.from_numpy(momenta).type(tensor_scalar_type)
    control_points_torch = torch.from_numpy(control_points).type(tensor_scalar_type)
    template_points = {key: torch.from_numpy(value).type(tensor_scalar_type)
                       for key, value in template.get_points().items()}
    template_data = {key: torch.from_numpy(value).type(tensor_scalar_type)
                     for key, value in template.get_data().items()}

    geodesic = Geodesic(dense_mode=dense_mode,
                        concentration_of_time_points=concentration_of_time_points, t0=t0,
                        kernel=deformation_kernel, shoot_kernel_type=shoot_kernel_type,
                        use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

    if t0 is None:
        logger.warning('Defaulting geodesic t0 to 1.')
        geodesic.t0 = 0.
    else:
        geodesic.t0 = t0

    if tmax == -float('inf'):
        logger.warning('Defaulting geodesic tmax to 1.')
        geodesic.tmax = 1.
    else:
        geodesic.tmax = tmax

    if tmin == float('inf'):
        logger.warning('Defaulting geodesic tmin to 0.')
        geodesic.tmin = 0.
    else:
        geodesic.tmin = tmin

    assert geodesic.tmax >= geodesic.t0, 'The max time {} for the shooting should be larger than t0 {}' \
        .format(geodesic.tmax, geodesic.t0)
    assert geodesic.tmin <= geodesic.t0, 'The min time for the shooting should be lower than t0.' \
        .format(geodesic.tmin, geodesic.t0)

    geodesic.set_control_points_t0(control_points_torch)
    geodesic.set_kernel(deformation_kernel)
    geodesic.set_use_rk2_for_shoot(use_rk2_for_shoot)
    geodesic.set_use_rk2_for_flow(use_rk2_for_flow)
    geodesic.set_template_points_t0(template_points)

    # Single momenta: single shooting
    if len(momenta.shape) == 2:
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.update()
        names = [elt for elt in t_name]
        geodesic.write('Shooting', names, t_name_extension, template, template_data, output_dir,
                       write_adjoint_parameters=True)

    # Several shootings to compute
    else:
        for i in range(len(momenta_torch)):
            geodesic.set_momenta_t0(momenta_torch[i])
            geodesic.update()
            names = [elt for elt in t_name]
            geodesic.write('Shooting' + "_" + str(i), names, t_name_extension, template, template_data, output_dir,
                           write_adjoint_parameters=True)
