import torch

from core.model_tools.manifolds.exponential_interface import ExponentialInterface

import logging
logger = logging.getLogger(__name__)


"""
Exponential on \R for 1/(q**2(1-q)) metric i.e. logistic curves.
"""

class LogisticExponential(ExponentialInterface):

    def __init__(self):
        # Mother class constructor
        ExponentialInterface.__init__(self)
        self.has_closed_form = True
        self.has_closed_form_parallel_transport = False

    def inverse_metric(self, q):
        return torch.diag((q*(1-q))**2)

    def closed_form(self, q, v, t):
        return 1./(1 + (1/q - 1) * torch.exp(-1.*v/(q * (1-q)) * t))

    def closed_form_velocity(self, q, v, t):
        aux = torch.exp(-1. * v * t / (q * (1 - q)))
        return v/q**2 * aux/(1 + (1/q - 1) * aux)**2

