import torch
import math

from core import default
from core.model_tools.deformations.exponential import Exponential
from core.model_tools.deformations.geodesic import Geodesic
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata
from support import utilities
from support import kernels as kernel_factory

import logging
logger = logging.getLogger(__name__)


def compute_parallel_transport(template_specifications,
                               dimension=default.dimension,
                               tensor_scalar_type=default.tensor_scalar_type,
                               tensor_integer_type=default.tensor_integer_type,

                               deformation_kernel_type=default.deformation_kernel_type,
                               deformation_kernel_width=default.deformation_kernel_width,

                               shoot_kernel_type=None,
                               initial_control_points=default.initial_control_points,
                               initial_momenta=default.initial_momenta,
                               initial_control_points_to_transport=default.initial_control_points_to_transport,
                               initial_momenta_to_transport=default.initial_momenta_to_transport,
                               tmin=default.tmin, tmax=default.tmax,
                               dense_mode=default.dense_mode,
                               concentration_of_time_points=default.concentration_of_time_points,
                               t0=default.t0,
                               number_of_time_points=default.number_of_time_points,
                               use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                               gpu_mode=default.gpu_mode,
                               output_dir=default.output_dir, **kwargs
                               ):
    deformation_kernel = kernel_factory.factory(deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=deformation_kernel_width)

    """
    Compute parallel transport
    """
    if initial_control_points is None:
        raise RuntimeError("Please provide initial control points")
    if initial_momenta is None:
        raise RuntimeError("Please provide initial momenta")
    if initial_momenta_to_transport is None:
        raise RuntimeError("Please provide initial momenta to transport")

    control_points = read_2D_array(initial_control_points)
    initial_momenta = read_3D_array(initial_momenta)
    initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)

    if initial_control_points_to_transport is None:
        logger.warning(
            "initial-control-points-to-transport was not specified, "
            "I am assuming they are the same as initial-control-points")
        control_points_to_transport = control_points
        need_to_project_initial_momenta = False
    else:
        control_points_to_transport = read_2D_array(initial_control_points_to_transport)
        need_to_project_initial_momenta = True

    device, _ = utilities.get_best_device(gpu_mode)

    control_points = utilities.move_data(control_points, dtype=tensor_scalar_type, device=device)
    control_points_to_transport = utilities.move_data(control_points_to_transport, dtype=tensor_scalar_type, device=device)
    initial_momenta = utilities.move_data(initial_momenta, dtype=tensor_scalar_type, device=device)
    initial_momenta_to_transport = utilities.move_data(initial_momenta_to_transport, dtype=tensor_scalar_type, device=device)

    # We start by projecting the initial momenta if they are not carried at the reference progression control points.
    if need_to_project_initial_momenta:
        velocity = deformation_kernel.convolve(control_points, control_points_to_transport, initial_momenta_to_transport)
        kernel_matrix = deformation_kernel.get_kernel_matrix(control_points)

        """
        The following code block needs to be done on cpu due to the high memory usage of the matrix inversion.
        TODO: maybe use Keops Inv ?
        """
        velocity = utilities.move_data(velocity, dtype=tensor_scalar_type, device='cpu')  # TODO: could this be done on gpu ?
        kernel_matrix = utilities.move_data(kernel_matrix, dtype=tensor_scalar_type, device='cpu')  # TODO: could this be done on gpu ?

        cholesky_kernel_matrix = torch.cholesky(kernel_matrix, upper=True)
        # cholesky_kernel_matrix = torch.Tensor(np.linalg.cholesky(kernel_matrix.data.numpy()).type_as(kernel_matrix))#Dirty fix if pytorch fails.
        projected_momenta = torch.cholesky_solve(velocity, cholesky_kernel_matrix, upper=True).squeeze().contiguous()

    else:
        projected_momenta = initial_momenta_to_transport

    """
    Re-send data to device depending on gpu_mode
    """
    device, _ = utilities.get_best_device(gpu_mode)
    projected_momenta = utilities.move_data(projected_momenta, dtype=tensor_scalar_type, device=device)

    """
    Second half of the code.
    """

    objects_list, objects_name, objects_name_extension, _, _ = create_template_metadata(template_specifications, dimension, gpu_mode=gpu_mode)
    template = DeformableMultiObject(objects_list)

    template_points = template.get_points()
    template_points = {key: utilities.move_data(value, dtype=tensor_scalar_type, device=device) for key, value in template_points.items()}

    template_data = template.get_data()
    template_data = {key: utilities.move_data(value, dtype=tensor_scalar_type, device=device) for key, value in template_data.items()}

    geodesic = Geodesic(dense_mode=dense_mode,
                        concentration_of_time_points=concentration_of_time_points, t0=t0,
                        kernel=deformation_kernel, shoot_kernel_type=shoot_kernel_type,
                        use_rk2_for_shoot=True, use_rk2_for_flow=use_rk2_for_flow)

    # Those are mandatory parameters.
    assert math.fabs(tmin) != float("inf"), "Please specify a minimum time for the geodesic trajectory"
    assert math.fabs(tmax) != float("inf"), "Please specify a maximum time for the geodesic trajectory"

    geodesic.set_tmin(tmin)
    geodesic.set_tmax(tmax)
    if t0 is None:
        geodesic.set_t0(geodesic.tmin)
    else:
        geodesic.set_t0(t0)

    geodesic.set_momenta_t0(initial_momenta)
    geodesic.set_control_points_t0(control_points)
    geodesic.set_template_points_t0(template_points)
    geodesic.update()

    # We write the flow of the geodesic
    geodesic.write("Regression", objects_name, objects_name_extension, template, template_data, output_dir=output_dir)

    # Now we transport!
    parallel_transport_trajectory = geodesic.parallel_transport(projected_momenta)

    # Getting trajectory caracteristics:
    times = geodesic.get_times()
    control_points_traj = geodesic.get_control_points_trajectory()
    momenta_traj = geodesic.get_momenta_trajectory()

    exponential = Exponential(dense_mode=dense_mode,
                              kernel=deformation_kernel, shoot_kernel_type=shoot_kernel_type,
                              number_of_time_points=number_of_time_points,
                              use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

    # We save the parallel trajectory
    for i, (time, cp, mom, transported_mom) in enumerate(
            zip(times, control_points_traj, momenta_traj, parallel_transport_trajectory)):
        # Writing the momenta/cps
        write_2D_array(cp.detach().cpu().numpy(), output_dir, "ControlPoints_tp_{0:d}__age_{1:.2f}.txt".format(i, time))
        write_3D_array(mom.detach().cpu().numpy(), output_dir, "Momenta_tp_{0:d}__age_{1:.2f}.txt".format(i, time))
        write_3D_array(transported_mom.detach().cpu().numpy(), output_dir,
                       "Transported_Momenta_tp_{0:d}__age_{1:.2f}.txt".format(i, time))

        deformed_points = geodesic.get_template_points(time)

        # Shooting from the geodesic:
        exponential.set_initial_template_points(deformed_points)
        exponential.set_initial_control_points(cp)
        exponential.set_initial_momenta(transported_mom)
        exponential.update()

        parallel_points = exponential.get_template_points()
        parallel_data = template.get_deformed_data(parallel_points, template_data)

        names = [
            objects_name[k] + "_parallel_curve_tp_{0:d}__age_{1:.2f}".format(i, time) + objects_name_extension[k]
            for k in range(len(objects_name))]

        template.write(output_dir, names, {key: value.detach().cpu().numpy() for key, value in parallel_data.items()})
