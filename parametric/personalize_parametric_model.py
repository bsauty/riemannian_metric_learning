import sys

sys.path.append('../riemannian_metric_learning')

from support.utilities.general_settings import Settings
import estimate_longitudinal_metric_model
from in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from in_out.dataset_functions import read_and_create_scalar_dataset 

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger.setLevel(logging.INFO)

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

    logger.info('Setting ouput directory to: ' + output_dir)
    deformetrica = dfca.Deformetrica(output_dir=output_dir, verbosity=logger.level)

    # logger.info('[ read_all_xmls function ]')
    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(model_xml_path, dataset_xml_path, optimization_parameters_xml_path)
    
    # Creating the dataset object
    dataset = read_and_create_scalar_dataset(xml_parameters)
    observation_type = 'scalar'
    
    """
    Gradient descent on the individual parameters 
    """

    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    #xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
    
    #xml_parameters.scale_initial_step_size = False
    xml_parameters.max_iterations = 50
    xml_parameters.max_line_search_iterations = 2
    
    xml_parameters.initial_step_size = 1
    xml_parameters.line_search_shrink = 0.5
    xml_parameters.line_search_expand = 1.1
    xml_parameters.save_every_n_iters = 1
    
    # Freezing some variances !
    xml_parameters.freeze_acceleration_variance = True
    xml_parameters.freeze_metric_parameters = True
    xml_parameters.freeze_noise_variance = True
    xml_parameters.freeze_onset_age_variance = True
    xml_parameters.freeze_reference_time = True
    
    # Freezing other variables
    xml_parameters.freeze_modulation_matrix = True
    xml_parameters.freeze_p0 = True
    xml_parameters.freeze_v0 = True
    xml_parameters.output_dir = output_dir
    Settings().output_dir = output_dir
    
    logger.info(" >>> Performing gradient descent.")
    
    estimate_longitudinal_metric_model(xml_parameters, logger=logger)

