import argparse
import logging
import sys

sys.path.append('../../riemannian_metric_learning')

from api import deformetrica as dfca
from support.utilities.general_settings import Settings
from in_out.xml_parameters import XmlParameters
from launch.estimate_longitudinal_metric_model import *

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    assert len(sys.argv) == 5, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> <output_folder> "

    model_xml_path = sys.argv[1]
    dataset_xml_path = sys.argv[2]
    optimization_parameters_xml_path = sys.argv[3]
    output_dir = sys.argv[4]

    # set logging level
    try:
        logger.setLevel('INFO')
    except ValueError:
        logger.warning('Logging level was not recognized. Using INFO.')
        logger.setLevel(logging.INFO)

    """
    Read xml files, set general settings, and call the adapted function.
    """

    logger.info('Setting output directory to: ' + output_dir)
    Settings().output_dir = output_dir

    deformetrica = dfca.Deformetrica(output_dir=output_dir, verbosity=logger.level)

    # logger.info('[ read_all_xmls function ]')
    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(model_xml_path, dataset_xml_path, optimization_parameters_xml_path)

    xml_parameters.freeze_p0 = False
    xml_parameters.freeze_v0 = False

    estimate_longitudinal_metric_model(xml_parameters, logger)
