import warnings
from copy import deepcopy
import torch

from core import default
from in_out.array_readers_and_writers import *
from support import utilities
from support import kernels as kernel_factory

import logging
logger = logging.getLogger(__name__)


class Exponential:
    """
    Control-point-based LDDMM exponential, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dense_mode=default.dense_mode,
                 kernel=default.deformation_kernel,
                 shoot_kernel_type=None,
                 number_of_time_points=None,
                 initial_control_points=None, control_points_t=None,
                 initial_momenta=None, momenta_t=None,
                 initial_template_points=None, template_points_t=None,
                 shoot_is_modified=True, flow_is_modified=True, use_rk2_for_shoot=False, use_rk2_for_flow=False):

        self.dense_mode = dense_mode
        self.kernel = kernel

        if shoot_kernel_type is not None:
            self.shoot_kernel = kernel_factory.factory(shoot_kernel_type, gpu_mode=kernel.gpu_mode, kernel_width=kernel.kernel_width)
        else:
            self.shoot_kernel = self.kernel

        # logger.debug(hex(id(self)) + ' using kernel: ' + str(self.kernel))
        # logger.debug(hex(id(self)) + ' using shoot_kernel: ' + str(self.shoot_kernel))

        self.number_of_time_points = number_of_time_points
        # Initial position of control points
        self.initial_control_points = initial_control_points
        # Control points trajectory
        self.control_points_t = control_points_t
        # Initial momenta
        self.initial_momenta = initial_momenta
        # Momenta trajectory
        self.momenta_t = momenta_t
        # Initial template points
        self.initial_template_points = initial_template_points
        # Trajectory of the whole vertices of landmark type at different time steps.
        self.template_points_t = template_points_t
        # If the cp or mom have been modified:
        self.shoot_is_modified = shoot_is_modified
        # If the template points has been modified
        self.flow_is_modified = flow_is_modified
        # Wether to use a RK2 or a simple euler for shooting or flowing respectively.
        self.use_rk2_for_shoot = use_rk2_for_shoot
        self.use_rk2_for_flow = use_rk2_for_flow
        # Contains the inverse kernel matrices for the time points 1 to self.number_of_time_points
        # (ACHTUNG does not contain the initial matrix, it is not needed)
        self.cometric_matrices = {}
        # self.cholesky_matrices = {}

    def move_data_to_(self, device):
        if self.initial_control_points is not None:
            self.initial_control_points = utilities.move_data(self.initial_control_points, device)
        if self.initial_momenta is not None:
            self.initial_momenta = utilities.move_data(self.initial_momenta, device)

        if self.initial_template_points is not None:
            self.initial_template_points = {key: utilities.move_data(value, device) for key, value in
                                            self.initial_template_points.items()}

    def light_copy(self):
        light_copy = Exponential(self.dense_mode,
                                 deepcopy(self.kernel), self.shoot_kernel.kernel_type,
                                 self.number_of_time_points,
                                 self.initial_control_points, self.control_points_t,
                                 self.initial_momenta, self.momenta_t,
                                 self.initial_template_points, self.template_points_t,
                                 self.shoot_is_modified, self.flow_is_modified,
                                 self.use_rk2_for_shoot, self.use_rk2_for_flow)
        return light_copy

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2_for_shoot(self, flag):
        self.shoot_is_modified = True
        self.use_rk2_for_shoot = flag

    def set_use_rk2_for_flow(self, flag):
        self.flow_is_modified = True
        self.use_rk2_for_flow = flag

    def get_kernel_type(self):
        return self.kernel.kernel_type

    def get_kernel_width(self):
        return self.kernel.kernel_width

    def set_kernel(self, kernel):
        # TODO which kernel to set ?
        self.kernel = kernel

    def set_initial_template_points(self, td):
        self.initial_template_points = td
        self.flow_is_modified = True

    def get_initial_template_points(self):
        return self.initial_template_points

    def set_initial_control_points(self, cps):
        self.shoot_is_modified = True
        self.initial_control_points = cps

    def get_initial_control_points(self):
        return self.initial_control_points

    def get_initial_momenta(self):
        return self.initial_momenta

    def set_initial_momenta(self, mom):
        self.shoot_is_modified = True
        self.initial_momenta = mom

    def get_initial_momenta(self):
        return self.initial_momenta

    def scalar_product(self, cp, mom1, mom2):
        """
        returns the scalar product 'mom1 K(cp) mom 2'
        """
        return torch.sum(mom1 * self.kernel.convolve(cp, cp, mom2))

    def get_template_points(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if self.flow_is_modified:
            msg = "You tried to get some template points, but the flow was modified. " \
                  "The exponential should be updated before."
            warnings.warn(msg)
        if time_index is None:
            return {key: self.template_points_t[key][-1] for key in self.initial_template_points.keys()}
        return {key: self.template_points_t[key][time_index] for key in self.initial_template_points.keys()}

    def get_norm_squared(self):
        return self.scalar_product(self.initial_control_points, self.initial_momenta, self.initial_momenta)

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call shoot or flow on the deformation.
        """
        assert self.number_of_time_points > 0
        if self.shoot_is_modified:
            self.cometric_matrices.clear()
            # self.cholesky_matrices.clear()
            self.shoot()
            if self.initial_template_points is not None:
                self.flow()
            elif not self.dense_mode:
                msg = "In exponential update, I am not flowing because I don't have any template points to flow"
                logger.warning(msg)

        if self.flow_is_modified:
            if self.initial_template_points is not None:
                self.flow()
            elif not self.dense_mode:
                msg = "In exponential update, I am not flowing because I don't have any template points to flow"
                logger.warning(msg)

    def shoot(self):
        """
        Computes the flow of momenta and control points.
        """
        # start_update = time.perf_counter()
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"

        # Integrate the Hamiltonian equations.
        self.control_points_t = [self.initial_control_points]
        self.momenta_t = [self.initial_momenta]

        dt = 1.0 / float(self.number_of_time_points - 1)

        if self.use_rk2_for_shoot:
            for i in range(self.number_of_time_points - 1):
                new_cp, new_mom = self._rk2_step(self.shoot_kernel, self.control_points_t[i], self.momenta_t[i], dt,
                                                 return_mom=True)
                self.control_points_t.append(new_cp)
                self.momenta_t.append(new_mom)

        else:
            for i in range(self.number_of_time_points - 1):
                new_cp, new_mom = self._euler_step(self.shoot_kernel, self.control_points_t[i], self.momenta_t[i], dt)
                self.control_points_t.append(new_cp)
                self.momenta_t.append(new_mom)

        # Correctly resets the attribute flag.
        self.shoot_is_modified = False
        # logger.info('exponential.shoot(): ' + str(time.perf_counter() - start_update))

    def flow(self):
        """
        Flow the trajectory of the landmark and/or image points.
        """
        # start_update = time.perf_counter()
        assert not self.shoot_is_modified, "CP or momenta were modified and the shoot not computed, and now you are asking me to flow ?"
        assert len(self.control_points_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"

        # Initialization.
        dt = 1.0 / float(self.number_of_time_points - 1)
        self.template_points_t = {}

        # Special case of the dense mode.
        if self.dense_mode:
            assert 'image_points' not in self.initial_template_points.keys(), 'Dense mode not allowed with image data.'
            self.template_points_t['landmark_points'] = self.control_points_t
            self.flow_is_modified = False
            return

        # Flow landmarks points.
        if 'landmark_points' in self.initial_template_points.keys():
            landmark_points = [self.initial_template_points['landmark_points']]

            for i in range(self.number_of_time_points - 1):
                d_pos = self.kernel.convolve(landmark_points[i], self.control_points_t[i], self.momenta_t[i])
                landmark_points.append(landmark_points[i] + dt * d_pos)

                if self.use_rk2_for_flow:
                    # In this case improved euler (= Heun's method)
                    # to save one computation of convolve gradient per iteration.
                    if i < self.number_of_time_points - 2:
                        landmark_points[-1] = landmark_points[i] + dt / 2 * \
                                              (self.kernel.convolve(landmark_points[i + 1],
                                                                    self.control_points_t[i + 1], self.momenta_t[i + 1]) + d_pos)
                    else:
                        final_cp, final_mom = self._rk2_step(self.kernel, self.control_points_t[-1], self.momenta_t[-1],
                                                             dt, return_mom=True)
                        landmark_points[-1] = landmark_points[i] + dt / 2 * (
                                self.kernel.convolve(landmark_points[i + 1], final_cp, final_mom) + d_pos)

            self.template_points_t['landmark_points'] = landmark_points

        # Flow image points.
        if 'image_points' in self.initial_template_points.keys():
            image_points = [self.initial_template_points['image_points']]

            dimension = self.initial_control_points.size(1)
            image_shape = image_points[0].size()

            for i in range(self.number_of_time_points - 1):
                vf = self.kernel.convolve(image_points[0].contiguous().view(-1, dimension), self.control_points_t[i],
                                          self.momenta_t[i]).view(image_shape)
                dY = self._compute_image_explicit_euler_step_at_order_1(image_points[i], vf)
                image_points.append(image_points[i] - dt * dY)

            if self.use_rk2_for_flow:
                msg = 'RK2 not implemented to flow image points.'
                logger.warning(msg)

            self.template_points_t['image_points'] = image_points

        assert len(self.template_points_t) > 0, 'That\'s unexpected'

        # Correctly resets the attribute flag.
        self.flow_is_modified = False
        # logger.info('exponential.flow(): ' + str(time.perf_counter() - start_update))

    def parallel_transport(self, momenta_to_transport, initial_time_point=0, is_orthogonal=False):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        if is_orthogonal is on, then the momenta to transport must be orthogonal to the momenta of the geodesic.
        Note: uses shoot kernel
        """

        # Sanity checks ------------------------------------------------------------------------------------------------
        assert not self.shoot_is_modified, "You want to parallel transport but the shoot was modified, please update."
        assert self.use_rk2_for_shoot, "The shoot integration must be done with a second order numerical scheme in order to use parallel transport."
        assert (momenta_to_transport.size() == self.initial_momenta.size())

        # Special cases, where the transport is simply the identity ----------------------------------------------------
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        if (torch.norm(self.initial_momenta).detach().cpu().numpy() < 1e-6 or
                torch.norm(momenta_to_transport).detach().cpu().numpy() < 1e-6):
            parallel_transport_t = [momenta_to_transport] * (self.number_of_time_points - initial_time_point)
            return parallel_transport_t

        # Step sizes ---------------------------------------------------------------------------------------------------
        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # For printing -------------------------------------------------------------------------------------------------
        worst_renormalization_factor = 1.0

        # Optional initial orthogonalization ---------------------------------------------------------------------------
        norm_squared = self.get_norm_squared()
        if not is_orthogonal:
            sp = self.scalar_product(self.control_points_t[initial_time_point], momenta_to_transport,
                                     self.momenta_t[initial_time_point]) / norm_squared
            momenta_to_transport_orthogonal = momenta_to_transport - sp * self.momenta_t[initial_time_point]
            parallel_transport_t = [momenta_to_transport_orthogonal]
        else:
            sp = (self.scalar_product(
                self.control_points_t[initial_time_point], momenta_to_transport,
                self.momenta_t[initial_time_point]) / norm_squared).detach().cpu().numpy()
            assert abs(sp) < 1e-2, \
                'Error: the momenta to transport is not orthogonal to the driving momenta, ' \
                'but the is_orthogonal flag is active. sp = %.3E' % sp
            parallel_transport_t = [momenta_to_transport]

        # Then, store the initial norm of this orthogonal momenta ------------------------------------------------------
        initial_norm_squared = self.scalar_product(self.control_points_t[initial_time_point], parallel_transport_t[0],
                                                   parallel_transport_t[0])

        for i in range(initial_time_point, self.number_of_time_points - 1):
            # Shoot the two perturbed geodesics ------------------------------------------------------------------------
            cp_eps_pos = self._rk2_step(self.shoot_kernel, self.control_points_t[i],
                                        self.momenta_t[i] + epsilon * parallel_transport_t[-1], h, return_mom=False)
            cp_eps_neg = self._rk2_step(self.shoot_kernel, self.control_points_t[i],
                                        self.momenta_t[i] - epsilon * parallel_transport_t[-1], h, return_mom=False)

            # Compute J/h ----------------------------------------------------------------------------------------------
            approx_velocity = (cp_eps_pos - cp_eps_neg) / (2 * epsilon * h)

            # We need to find the cotangent space version of this vector -----------------------------------------------
            # If we don't have already the cometric matrix, we compute and store it.
            # TODO: add optional flag for not saving this if it's too large.
            # OPTIM: keep an eye on https://github.com/pytorch/pytorch/issues/4669
            if i not in self.cometric_matrices:
                kernel_matrix = self.shoot_kernel.get_kernel_matrix(self.control_points_t[i + 1])
                # self.kernel_matrices[i] = kernel_matrix
                # self.cholesky_matrices[i] = torch.potrf(kernel_matrix.t().matmul(kernel_matrix), upper=False)
                # self.cholesky_matrices[i] = torch.cholesky(kernel_matrix, upper=False)
                # self.cholesky_matrices[i] = torch.potrf(kernel_matrix, upper=True)
                self.cometric_matrices[i] = torch.inverse(kernel_matrix)
                # self.cometric_matrices[i] = torch.inverse(kernel_matrix.cuda()).cpu()

            # Solve the linear system.
            # rhs = approx_velocity.matmul(self.kernel_matrices[i])
            # z = torch.trtrs(rhs.t(), self.cholesky_matrices[i], transpose=False, upper=False)[0]
            # approx_momenta = torch.trtrs(z, self.cholesky_matrices[i], transpose=True, upper=False)[0]
            # approx_momenta = torch.potrs(approx_velocity, self.cholesky_matrices[i], upper=True)
            approx_momenta = torch.mm(self.cometric_matrices[i], approx_velocity)

            # We get rid of the component of this momenta along the geodesic velocity:
            scalar_prod_with_velocity = self.scalar_product(self.control_points_t[i + 1], approx_momenta,
                                                            self.momenta_t[i + 1]) / norm_squared

            approx_momenta = approx_momenta - scalar_prod_with_velocity * self.momenta_t[i + 1]

            # Renormalization ------------------------------------------------------------------------------------------
            approx_momenta_norm_squared = self.scalar_product(self.control_points_t[i + 1], approx_momenta,
                                                              approx_momenta)

            renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
            renormalized_momenta = approx_momenta * renormalization_factor

            if abs(renormalization_factor.detach().cpu().numpy() - 1.) > 0.1:
                raise ValueError('Absurd required renormalization factor during parallel transport: %.4f. '
                                 'Exception raised.' % renormalization_factor.detach().cpu().numpy())
            elif abs(renormalization_factor.detach().cpu().numpy() - 1.) > abs(worst_renormalization_factor - 1.):
                worst_renormalization_factor = renormalization_factor.detach().cpu().numpy()

            # Finalization ---------------------------------------------------------------------------------------------
            parallel_transport_t.append(renormalized_momenta)

        assert len(parallel_transport_t) == self.number_of_time_points - initial_time_point, \
            "Oops, something went wrong."

        # We now need to add back the component along the velocity to the transported vectors.
        if not is_orthogonal:
            parallel_transport_t = [parallel_transport_t[i] + sp * self.momenta_t[i]
                                    for i in range(initial_time_point, self.number_of_time_points)]

        if abs(worst_renormalization_factor - 1.) > 0.05:
            msg = ("Watch out, a large renormalization factor %.4f is required during the parallel transport. "
                   "Try using a finer discretization." % worst_renormalization_factor)
            logger.warning(msg)

        return parallel_transport_t

    ####################################################################################################################
    ### Extension methods:
    ####################################################################################################################

    def extend(self, number_of_additional_time_points):

        # Special case of the exponential reduced to a single point.
        if self.number_of_time_points == 1:
            self.number_of_time_points += number_of_additional_time_points
            self.update()
            return

        # Extended shoot.
        dt = 1.0 / float(self.number_of_time_points - 1)  # Same time-step.
        for i in range(number_of_additional_time_points):
            if self.use_rk2_for_shoot:
                new_cp, new_mom = self._rk2_step(self.kernel, self.control_points_t[-1], self.momenta_t[-1], dt,
                                                 return_mom=True)
            else:
                new_cp, new_mom = self._euler_step(self.kernel, self.control_points_t[-1], self.momenta_t[-1], dt)

            self.control_points_t.append(new_cp)
            self.momenta_t.append(new_mom)

        # Extended flow.
        # Special case of the dense mode.
        if self.dense_mode:
            assert 'image_points' not in self.initial_template_points.keys(), 'Dense mode not allowed with image data.'
            self.template_points_t['landmark_points'] = self.control_points_t

        # Standard case.
        else:
            # Flow landmark points.
            if 'landmark_points' in self.initial_template_points.keys():
                for ii in range(number_of_additional_time_points):
                    i = len(self.template_points_t['landmark_points']) - 1
                    d_pos = self.kernel.convolve(self.template_points_t['landmark_points'][i], self.control_points_t[i],
                                                 self.momenta_t[i])
                    self.template_points_t['landmark_points'].append(
                        self.template_points_t['landmark_points'][i] + dt * d_pos)

                    if self.use_rk2_for_flow:
                        # In this case improved euler (= Heun's method) to save one computation of convolve gradient.
                        self.template_points_t['landmark_points'][i + 1] = (
                            self.template_points_t['landmark_points'][i] + dt / 2 * (self.kernel.convolve(
                            self.template_points_t['landmark_points'][i + 1], self.control_points_t[i + 1],
                            self.momenta_t[i + 1]) + d_pos))

            # Flow image points.
            if 'image_points' in self.initial_template_points.keys():
                dimension = self.initial_control_points.size(1)
                image_shape = self.initial_template_points['image_points'].size()

                for ii in range(number_of_additional_time_points):
                    i = len(self.template_points_t['image_points']) - 1
                    vf = self.kernel.convolve(self.initial_template_points['image_points'].contiguous().view(-1, dimension),
                                              self.control_points_t[i], self.momenta_t[i]).view(image_shape)
                    dY = self._compute_image_explicit_euler_step_at_order_1(self.template_points_t['image_points'][i], vf)
                    self.template_points_t['image_points'].append(self.template_points_t['image_points'][i] - dt * dY)

                if self.use_rk2_for_flow:
                    msg = 'RK2 not implemented to flow image points.'
                    logger.warning(msg)

        # Scaling of the new length.
        length_ratio = float(self.number_of_time_points + number_of_additional_time_points - 1) \
                       / float(self.number_of_time_points - 1)
        self.number_of_time_points += number_of_additional_time_points
        self.initial_momenta = self.initial_momenta * length_ratio
        self.momenta_t = [elt * length_ratio for elt in self.momenta_t]

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    @staticmethod
    def _euler_step(kernel, cp, mom, h):
        """
        simple euler step of length h, with cp and mom. It always returns mom.
        """
        assert cp.device == mom.device, 'tensors must be on the same device, cp.device=' + str(
            cp.device) + ', mom.device=' + str(mom.device)
        return cp + h * kernel.convolve(cp, cp, mom), \
               mom - h * kernel.convolve_gradient(mom, cp)

    @staticmethod
    def _rk2_step(kernel, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
        also used in parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        assert cp.device == mom.device, 'tensors must be on the same device, cp.device=' + str(
            cp.device) + ', mom.device=' + str(mom.device)

        mid_cp = cp + h / 2. * kernel.convolve(cp, cp, mom)
        mid_mom = mom - h / 2. * kernel.convolve_gradient(mom, cp)
        if return_mom:
            return cp + h * kernel.convolve(mid_cp, mid_cp, mid_mom), \
                   mom - h * kernel.convolve_gradient(mid_mom, mid_cp)
        else:
            return cp + h * kernel.convolve(mid_cp, mid_cp, mid_mom)

    # TODO. Wrap pytorch of an efficient C code ? Use keops ? Called ApplyH in PyCa. Check Numba as well.
    # @jit(parallel=True)
    @staticmethod
    def _compute_image_explicit_euler_step_at_order_1(Y, vf):
        assert Y.device == vf.device, 'tensors must be on the same device, Y.device=' + str(
            Y.device) + ', vf.device=' + str(vf.device)

        dY = torch.zeros(Y.shape, dtype=vf.dtype, device=vf.device)
        dimension = len(Y.shape) - 1

        if dimension == 2:
            ni, nj = Y.shape[:2]

            # Center.
            dY[1:ni - 1, :] = dY[1:ni - 1, :] + 0.5 * vf[1:ni - 1, :, 0].view(ni - 2, nj, 1).expand(ni - 2, nj, 2) * (
                    Y[2:ni, :] - Y[0:ni - 2, :])
            dY[:, 1:nj - 1] = dY[:, 1:nj - 1] + 0.5 * vf[:, 1:nj - 1, 1].view(ni, nj - 2, 1).expand(ni, nj - 2, 2) * (
                    Y[:, 2:nj] - Y[:, 0:nj - 2])

            # Borders.
            dY[0, :] = dY[0, :] + vf[0, :, 0].view(nj, 1).expand(nj, 2) * (Y[1, :] - Y[0, :])
            dY[ni - 1, :] = dY[ni - 1, :] + vf[ni - 1, :, 0].view(nj, 1).expand(nj, 2) * (Y[ni - 1, :] - Y[ni - 2, :])

            dY[:, 0] = dY[:, 0] + vf[:, 0, 1].view(ni, 1).expand(ni, 2) * (Y[:, 1] - Y[:, 0])
            dY[:, nj - 1] = dY[:, nj - 1] + vf[:, nj - 1, 1].view(ni, 1).expand(ni, 2) * (Y[:, nj - 1] - Y[:, nj - 2])

        elif dimension == 3:
            ni, nj, nk = Y.shape[:3]

            # Center.
            dY[1:ni - 1, :, :] = dY[1:ni - 1, :, :] + 0.5 * vf[1:ni - 1, :, :, 0].view(ni - 2, nj, nk, 1).expand(ni - 2,
                                                                                                                 nj, nk,
                                                                                                                 3) * (
                                         Y[2:ni, :, :] - Y[0:ni - 2, :, :])
            dY[:, 1:nj - 1, :] = dY[:, 1:nj - 1, :] + 0.5 * vf[:, 1:nj - 1, :, 1].view(ni, nj - 2, nk, 1).expand(ni,
                                                                                                                 nj - 2,
                                                                                                                 nk,
                                                                                                                 3) * (
                                         Y[:, 2:nj, :] - Y[:, 0:nj - 2, :])
            dY[:, :, 1:nk - 1] = dY[:, :, 1:nk - 1] + 0.5 * vf[:, :, 1:nk - 1, 2].view(ni, nj, nk - 2, 1).expand(ni, nj,
                                                                                                                 nk - 2,
                                                                                                                 3) * (
                                         Y[:, :, 2:nk] - Y[:, :, 0:nk - 2])

            # Borders.
            dY[0, :, :] = dY[0, :, :] + vf[0, :, :, 0].view(nj, nk, 1).expand(nj, nk, 3) * (Y[1, :, :] - Y[0, :, :])
            dY[ni - 1, :, :] = dY[ni - 1, :, :] + vf[ni - 1, :, :, 0].view(nj, nk, 1).expand(nj, nk, 3) * (
                    Y[ni - 1, :, :] - Y[ni - 2, :, :])

            dY[:, 0, :] = dY[:, 0, :] + vf[:, 0, :, 1].view(ni, nk, 1).expand(ni, nk, 3) * (Y[:, 1, :] - Y[:, 0, :])
            dY[:, nj - 1, :] = dY[:, nj - 1, :] + vf[:, nj - 1, :, 1].view(ni, nk, 1).expand(ni, nk, 3) * (
                    Y[:, nj - 1, :] - Y[:, nj - 2, :])

            dY[:, :, 0] = dY[:, :, 0] + vf[:, :, 0, 2].view(ni, nj, 1).expand(ni, nj, 3) * (Y[:, :, 1] - Y[:, :, 0])
            dY[:, :, nk - 1] = dY[:, :, nk - 1] + vf[:, :, nk - 1, 2].view(ni, nj, 1).expand(ni, nj, 3) * (
                    Y[:, :, nk - 1] - Y[:, :, nk - 2])

        else:
            raise RuntimeError('Invalid dimension of the ambient space: %d' % dimension)

        return dY

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write_flow(self, objects_names, objects_extensions, template, template_data, output_dir,
                   write_adjoint_parameters=False):

        assert not self.flow_is_modified, \
            "You are trying to write data relative to the flow, but it has been modified and not updated."

        for j in range(self.number_of_time_points):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for k, elt in enumerate(objects_names):
                names.append(elt + "__tp_" + str(j) + objects_extensions[k])

            deformed_points = self.get_template_points(j)
            deformed_data = template.get_deformed_data(deformed_points, template_data)
            template.write(output_dir, names,
                           {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

            if write_adjoint_parameters:
                cp = self.control_points_t[j].detach().cpu().numpy()
                mom = self.momenta_t[j].detach().cpu().numpy()
                write_2D_array(cp, output_dir, elt + "__ControlPoints__tp_" + str(j) + ".txt")
                write_3D_array(mom, output_dir, elt + "__Momenta__tp_" + str(j) + ".txt")

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert not self.shoot_is_modified, \
            "You are trying to write data relative to the shooting, but it has been modified and not updated."
        assert len(self.control_points_t) == len(self.momenta_t), \
            "Something is wrong, not as many cp as momenta in diffeo"
        for j, (control_points, momenta) in enumerate(zip(self.control_points_t, self.momenta_t)):
            write_2D_array(control_points.detach().cpu().numpy(), name + "__control_points_" + str(j) + ".txt")
            write_2D_array(momenta.detach().cpu().numpy(), name + "__momenta_" + str(j) + ".txt")
