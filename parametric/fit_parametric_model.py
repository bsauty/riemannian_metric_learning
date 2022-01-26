import argparse
import logging
import sys

sys.path.append('../../riemannian_metric_learning')

from api import deformetrica as dfca
from support.utilities.general_settings import Settings
from in_out.xml_parameters import XmlParameters
from launch.estimate_longitudinal_metric_model import *

logger = logging.getLogger(__name__)

path = 'tadpole/'

args = {'command':'estimate', 'verbosity':'INFO', 'output':'output',
        'model':path+'model_after_initialization.xml', 'dataset':path+'data_set.xml', 'parameters':path+'optimization_parameters_saem.xml'}

 # set logging level
try:
    logger.setLevel(args['verbosity'])
except ValueError:
    logger.warning('Logging level was not recognized. Using INFO.')
    logger.setLevel(logging.INFO)

"""
Read xml files, set general settings, and call the adapted function.
"""

logger.info('Setting output directory to: ' + args['output'])
output_dir = args['output']
Settings().output_dir = output_dir

deformetrica = dfca.Deformetrica(output_dir=output_dir, verbosity=logger.level)

# logger.info('[ read_all_xmls function ]')
xml_parameters = XmlParameters()
xml_parameters.read_all_xmls(args['model'],
                             args['dataset'] if args['command'] == 'estimate' else None,
                             args['parameters'])

xml_parameters.freeze_p0 = False
xml_parameters.freeze_v0 = False

estimate_longitudinal_metric_model(xml_parameters, logger)
