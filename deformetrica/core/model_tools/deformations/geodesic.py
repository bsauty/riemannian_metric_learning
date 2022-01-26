import time
import warnings


from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....in_out.array_readers_and_writers import *
from ....support import utilities

import logging
logger = logging.getLogger(__name__)


def _parallel_transport(*args):

    # read args
    compute_backward, exponential, momenta_to_transport_t0, is_orthogonal = args

    # compute
    if compute_backward:
        return compute_backward, exponential.parallel_transport(momenta_to_transport_t0, is_orthogonal=is_orthogonal)
    else:
        return compute_backward, exponential.parallel_transport(momenta_to_transport_t0, is_orthogonal=is_orthogonal)


class Geodesic:
    """
    Control-point-based LDDMM geodesic.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dense_mode=default.dense_mode,
                 kernel=default.deformation_kernel, shoot_kernel_type=None,
                 t0=default.t0, concentration_of_time_points=default.concentration_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow):

        self.concentration_of_time_points = concentration_of_time_points
        self.t0 = t0
        self.tmax = None
        self.tmin = None

        self.control_points_t0 = None
        self.momenta_t0 = None
        self.template_points_t0 = None

        self.backward_exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel, shoot_kernel_type=shoot_kernel_type,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        self.forward_exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel, shoot_kernel_type=shoot_kernel_type,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        # Flags to save extra computations that have already been made in the update methods.
        self.shoot_is_modified = True
        self.flow_is_modified = True
        self.backward_extension = 0
        self.forward_extension = 0

        # mp.set_sharing_strategy('file_system')
        # self.parallel_transport_pool = mp.Pool(processes=1)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2_for_shoot(self, flag):
        self.backward_exponential.set_use_rk2_for_shoot(flag)
        self.forward_exponential.set_use_rk2_for_shoot(flag)

    def set_use_rk2_for_flow(self, flag):
        self.backward_exponential.set_use_rk2_for_flow(flag)
        self.forward_exponential.set_use_rk2_for_flow(flag)

    def get_kernel_type(self):
        return self.backward_exponential.get_kernel_type()

    def set_kernel(self, kernel):
        self.backward_exponential.kernel = kernel
        self.forward_exponential.kernel = kernel

    def set_t0(self, t0):
        self.t0 = t0
        self.shoot_is_modified = True

    def get_tmin(self):
        return self.tmin

    def set_tmin(self, tmin, optimize=False):
        if not optimize:
            self.tmin = tmin
            self.shoot_is_modified = True

        else:
            if self.tmin is None:
                self.tmin = tmin

            elif tmin < self.tmin:
                if self.backward_exponential.number_of_time_points > 1:
                    dt = (self.t0 - self.tmin) / float(self.backward_exponential.number_of_time_points - 1)
                    self.backward_extension = int((self.tmin - tmin) / dt)
                    self.tmin -= self.backward_extension * dt
                else:
                    self.tmin = tmin
                    length = self.t0 - self.tmin
                    self.backward_extension = max(0, int(length * self.concentration_of_time_points + 0.5))
                    self.backward_exponential.set_initial_momenta(- self.momenta_t0 * length)

    def get_tmax(self):
        return self.tmax

    def set_tmax(self, tmax, optimize=False):
        if not optimize:
            self.tmax = tmax
            self.shoot_is_modified = True

        else:
            if self.tmax is None:
                self.tmax = tmax

            elif tmax > self.tmax:
                if self.forward_exponential.number_of_time_points > 1:
                    dt = (self.tmax - self.t0) / float(self.forward_exponential.number_of_time_points - 1)
                    self.forward_extension = int((tmax - self.tmax) / dt)
                    self.tmax += self.forward_extension * dt
                else:
                    self.tmax = tmax
                    length = self.tmax - self.t0
                    self.forward_extension = max(0, int(length * self.concentration_of_time_points + 0.5))
                    self.forward_exponential.set_initial_momenta(self.momenta_t0 * length)

    def get_template_points_t0(self):
        return self.template_points_t0

    def set_template_points_t0(self, td):
        self.template_points_t0 = td
        self.flow_is_modified = True

    def set_control_points_t0(self, cp):
        self.control_points_t0 = cp
        self.shoot_is_modified = True

    def set_momenta_t0(self, mom):
        self.momenta_t0 = mom
        self.shoot_is_modified = True

    def get_template_points(self, time):
        """
        Returns the position of the landmark points, at the given time.
        Performs a linear interpolation between the two closest available data points.
        """

        assert self.tmin <= time <= self.tmax
        if self.shoot_is_modified or self.flow_is_modified:
            msg = "Asking for deformed template data but the geodesic was modified and not updated"
            warnings.warn(msg)

        times = self.get_times()

        # Deal with the special case of a geodesic reduced to a single point.
        if len(times) == 1:
            logger.info('>> The geodesic seems to be reduced to a single point.')
            return self.template_points_t0

        # Standard case.
        for j in range(1, len(times)):
            if time - times[j] < 0: break

        # j = np.searchsorted(times[:-1], time, side='right')

        # if time <= self.t0:
        #     dt = (self.t0 - self.tmin) / (self.backward_exponential.number_of_time_points - 1)
        #     j = int((time-self.tmin)/dt) + 1
        #
        # else:
        #     dt = (self.tmax - self.t0) / (self.forward_exponential.number_of_time_points - 1)
        #     j = min(len(times)-1,
        #             int((time - self.t0) / dt) + self.backward_exponential.number_of_time_points)
        #
        # assert times[j-1] <= time
        # assert times[j] >= time

        device, _ = utilities.get_best_device(self.backward_exponential.kernel.gpu_mode)

        weight_left = utilities.move_data([(times[j] - time) / (times[j] - times[j - 1])], device=device, dtype=self.momenta_t0.dtype)
        weight_right = utilities.move_data([(time - times[j - 1]) / (times[j] - times[j - 1])], device=device, dtype=self.momenta_t0.dtype)
        template_t = {key: [utilities.move_data(v, device=device) for v in value] for key, value in self.get_template_points_trajectory().items()}

        deformed_points = {key: weight_left * value[j - 1] + weight_right * value[j]
                           for key, value in template_t.items()}

        return deformed_points

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """

        assert self.t0 >= self.tmin, "tmin should be smaller than t0"
        assert self.t0 <= self.tmax, "tmax should be larger than t0"

        if self.shoot_is_modified or self.flow_is_modified:

            device, _ = utilities.get_best_device(self.backward_exponential.kernel.gpu_mode)
            # Backward exponential -------------------------------------------------------------------------------------
            length = self.t0 - self.tmin
            self.backward_exponential.number_of_time_points = \
                max(1, int(length * self.concentration_of_time_points + 1.5))
            if self.shoot_is_modified:
                self.backward_exponential.set_initial_momenta(- self.momenta_t0 * length)
                self.backward_exponential.set_initial_control_points(self.control_points_t0)
            if self.flow_is_modified:
                self.backward_exponential.set_initial_template_points(self.template_points_t0)
            if self.backward_exponential.number_of_time_points > 1:
                self.backward_exponential.move_data_to_(device=device)
                self.backward_exponential.update()

            # Forward exponential --------------------------------------------------------------------------------------
            length = self.tmax - self.t0
            self.forward_exponential.number_of_time_points = \
                max(1, int(length * self.concentration_of_time_points + 1.5))
            if self.shoot_is_modified:
                self.forward_exponential.set_initial_momenta(self.momenta_t0 * length)
                self.forward_exponential.set_initial_control_points(self.control_points_t0)
            if self.flow_is_modified:
                self.forward_exponential.set_initial_template_points(self.template_points_t0)
            if self.forward_exponential.number_of_time_points > 1:
                self.forward_exponential.move_data_to_(device=device)
                self.forward_exponential.update()

            self.shoot_is_modified = False
            self.flow_is_modified = False
            self.backward_extension = 0
            self.forward_extension = 0

        else:
            if self.backward_extension > 0:
                self.backward_exponential.extend(self.backward_extension)
                self.backward_extension = 0

            if self.forward_extension > 0:
                self.forward_exponential.extend(self.forward_extension)
                self.forward_extension = 0

    def get_norm_squared(self):
        """
        Get the norm of the geodesic.
        """
        return self.forward_exponential.scalar_product(self.control_points_t0, self.momenta_t0, self.momenta_t0)

    def parallel_transport(self, momenta_to_transport_t0, is_orthogonal=False):
        """
        :param momenta_to_transport_t0: the vector to parallel transport, given at t0 and carried at control_points_t0
        :returns: the full trajectory of the parallel transport, from tmin to tmax.
        """
        start = time.perf_counter()

        if self.shoot_is_modified:
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)

        # backwards
        if self.backward_exponential.number_of_time_points > 1:
            backward_transport = self.backward_exponential.parallel_transport(momenta_to_transport_t0,
                                                                              is_orthogonal=is_orthogonal)
        else:
            backward_transport = [momenta_to_transport_t0]

        # forwards
        if self.forward_exponential.number_of_time_points > 1:
            forward_transport = self.forward_exponential.parallel_transport(momenta_to_transport_t0,
                                                                            is_orthogonal=is_orthogonal)
        else:
            forward_transport = []

        logger.debug('time taken to compute parallel_transport: ' + str(time.perf_counter() - start))
        assert backward_transport is not None
        assert forward_transport is not None
        return backward_transport[::-1] + forward_transport[1:]

    ####################################################################################################################
    ### Extension methods:
    ####################################################################################################################

    def extend_parallel_transport(self, parallel_transport_t, backward_extension, forward_extension,
                                  is_orthogonal=False):

        parallel_transport_t_backward_extension = [parallel_transport_t[0]]
        if backward_extension > 0:
            parallel_transport_t_backward_extension = self.backward_exponential.parallel_transport(
                parallel_transport_t_backward_extension[0],
                initial_time_point=self.backward_exponential.number_of_time_points - backward_extension - 1,
                is_orthogonal=is_orthogonal)

        parallel_transport_t_forward_extension = [parallel_transport_t[-1]]
        if forward_extension > 0:
            parallel_transport_t_forward_extension = self.forward_exponential.parallel_transport(
                parallel_transport_t_forward_extension[0],
                initial_time_point=self.forward_exponential.number_of_time_points - forward_extension - 1,
                is_orthogonal=is_orthogonal)

        parallel_transport_t = parallel_transport_t_backward_extension[:0:-1] \
                               + parallel_transport_t + parallel_transport_t_forward_extension[1:]
        return parallel_transport_t

    def get_times(self):
        times_backward = [self.t0]
        if self.backward_exponential.number_of_time_points > 1:
            times_backward = np.linspace(
                self.t0, self.tmin, num=self.backward_exponential.number_of_time_points).tolist()

        times_forward = [self.t0]
        if self.forward_exponential.number_of_time_points > 1:
            times_forward = np.linspace(
                self.t0, self.tmax, num=self.forward_exponential.number_of_time_points).tolist()

        return times_backward[::-1] + times_forward[1:]

    def get_control_points_trajectory(self):
        if self.shoot_is_modified:
            msg = "Trying to get cp trajectory in a non updated geodesic."
            warnings.warn(msg)

        backward_control_points_t = [self.backward_exponential.get_initial_control_points()]
        if self.backward_exponential.number_of_time_points > 1:
            backward_control_points_t = self.backward_exponential.control_points_t

        forward_control_points_t = [self.forward_exponential.get_initial_control_points()]
        if self.forward_exponential.number_of_time_points > 1:
            forward_control_points_t = self.forward_exponential.control_points_t

        return backward_control_points_t[::-1] + forward_control_points_t[1:]

    def get_momenta_trajectory(self):
        if self.shoot_is_modified:
            msg = "Trying to get mom trajectory in non updated geodesic."
            warnings.warn(msg)

        backward_momenta_t = [self.momenta_t0]
        if self.backward_exponential.number_of_time_points > 1:
            backward_length = self.tmin - self.t0
            backward_momenta_t = self.backward_exponential.momenta_t
            backward_momenta_t = [elt / backward_length for elt in backward_momenta_t]

        forward_momenta_t = [self.momenta_t0]
        if self.forward_exponential.number_of_time_points > 1:
            forward_length = self.tmax - self.t0
            forward_momenta_t = self.forward_exponential.momenta_t
            forward_momenta_t = [elt / forward_length for elt in forward_momenta_t]

        return backward_momenta_t[::-1] + forward_momenta_t[1:]

    def get_template_points_trajectory(self):
        if self.shoot_is_modified or self.flow_is_modified:
            msg = "Trying to get template trajectory in non updated geodesic."
            warnings.warn(msg)

        template_t = {}
        for key in self.template_points_t0.keys():

            backward_template_t = [self.backward_exponential.get_initial_template_points()[key]]
            if self.backward_exponential.number_of_time_points > 1:
                backward_template_t = self.backward_exponential.template_points_t[key]

            forward_template_t = [self.forward_exponential.get_initial_template_points()[key]]
            if self.forward_exponential.number_of_time_points > 1:
                forward_template_t = self.forward_exponential.template_points_t[key]

            template_t[key] = backward_template_t[::-1] + forward_template_t[1:]

        return template_t

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, root_name, objects_name, objects_extension, template, template_data, output_dir,
              write_adjoint_parameters=False):

        # Core loop ----------------------------------------------------------------------------------------------------
        times = self.get_times()
        for t, time in enumerate(times):
            names = []
            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                name = root_name + '__GeodesicFlow__' + object_name + '__tp_' + str(t) \
                       + ('__age_%.2f' % time) + object_extension
                names.append(name)
            deformed_points = self.get_template_points(time)
            deformed_data = template.get_deformed_data(deformed_points, template_data)
            template.write(output_dir, names,
                           {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

        # Optional writing of the control points and momenta -----------------------------------------------------------
        if write_adjoint_parameters:
            control_points_t = [elt.detach().cpu().numpy() for elt in self.get_control_points_trajectory()]
            momenta_t = [elt.detach().cpu().numpy() for elt in self.get_momenta_trajectory()]
            for t, (time, control_points, momenta) in enumerate(zip(times, control_points_t, momenta_t)):
                write_2D_array(control_points, output_dir, root_name + '__GeodesicFlow__ControlPoints__tp_' + str(t)
                               + ('__age_%.2f' % time) + '.txt')
                write_2D_array(momenta, output_dir, root_name + '__GeodesicFlow__Momenta__tp_' + str(t)
                               + ('__age_%.2f' % time) + '.txt')
