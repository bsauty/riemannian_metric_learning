import numpy as np
import pandas as pd
from leaspy import IndividualParameters
from leaspy.utils.posterior_analysis.abnormality import get_age_at_abnormality_conversion
    
    
def normalize(X):
    X -= X.mean()
    X /= X.std()
    return X


def get_reparametrized_ages(ages, individual_parameters, leaspy):
    r"""
                    
    Reparametrize the real ages of the patients onto the pathological timeline
    Parameters
    ----------
    individual_parameters: Individual parameters object
        Contains the individual parameters for each patient
    ages: dict {patient_idx: [ages]}
        Contains the patient ages to reparametrized
    leaspy: Leaspy object
        Contains the model parameters
    Returns
    -------
    reparametrized_ages: dict {patient_idx: [reparametrized_ages]}
        Contains the reparametrized ages
    """
    tau_mean = leaspy.model.parameters['tau_mean']
    indices = individual_parameters._indices
    reparametrized_ages = {}
    for idx, ages in ages.items():
        if idx not in indices:
            raise ValueError(f'The index {idx} is not in the individual parameters')
        idx_ip = individual_parameters[idx]
        alpha = np.exp(idx_ip['xi'])
        tau = idx_ip['tau']
        reparam_ages = [alpha * (age - tau ) + tau_mean for age in ages]
        reparametrized_ages[idx] = [_.numpy().tolist() for _ in reparam_ages]
    return reparametrized_ages



def plot_atlas_values(atlas_values_list, output_sig_map,
                     input_image_atlas='../data/AAL2.nii.gz'):
    """
        ###################################################################################
        ############## input is the list and then write the p-value for each tract and write them into a volume
        ###################################################################################
        This is a function to gave the regional P-value for each tract in the JHU_tracts_dispaly atals and save it as a nii.gz file.
        :param input_image_atlas: This is the atlas that you chosse should be a 3-dimension image
        :param twentyone_p_value_list: This is a list for each tracts in JHU_tracts_dispaly atals, in total 21, the first one is unknow, the last 20 are all the tracts
        :param output_sig_map: This is path for the destination significance map that you wanna save
        :param effect_size_img: default is False, to save the image for significant p-value, not effect size
        :return:

        Note: the p-value should be already corrected.
                To display the sig tracts in freeview, run : freeview FS_JHU_MD_uncor_sig.nii.gz /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
        """
    reading_input_image = nib.load(input_image_atlas)
    image = reading_input_image.get_data()
    image = np.array(image,dtype='f')

    labels = list(set(image.ravel()))
    print(f"There are {len(labels)} tracts in this atlas")

    output_image = np.array(image, dtype='f')

    for index, n in enumerate(labels):
        indice = np.array(np.where(image == n))
        print(f"The index of this tract is: {indice}")

        values = atlas_values_list[index]
        output_image[indice[0, :], indice[1, :], indice[2, :]] = values
        print(f"The pvalue of this tract is: {values}")

    output_image = nib.Nifti1Image(output_image, nib.load(input_image_atlas).get_affine())
    output_image.to_filename(output_sig_map)

    return output_sig_map


def predict_scores(data, ind, leaspy):
    """
    returns predicted values at the timepoints of the original data
    args:
    - data : original data (dataframe)
    - ind : individual_parameters (dataframe)
    - leaspy : leaspy object already fitted
    """
    # First of all, we need to group all sources into one single feature
    columns = ind.columns[2:]
    labels = leaspy.model.features

    ind_est = ind[['tau', 'xi']]
    if 'univariate' not in leaspy.model.name:
        ind_est['sources'] = ''
        for sub in ind.index:
            ind_est['sources'][sub] = [ind.loc[sub][source] for source in columns]

    # Then prepare the list of timepoints at which we want to estimate the values
    timepoints = dict(data.loc[ind.index]['TIME'])

    for sub in ind.index:
        if type(timepoints[sub]) == np.float64:
            timepoints[sub] = [timepoints[sub]]
        else:
            timepoints[sub] = list(data.loc[sub]['TIME'])


    # We estimate the values predicted by our leaspy model
    predicted_dict = leaspy.estimate(timepoints,ind_est.T)

    # Then we want to format it properly in a df to be able to compare with the real values
    predicted_df = pd.DataFrame(predicted_dict.items(), columns=['ID', 'sessions']).set_index('ID')

    df_lst = []

    for sub in predicted_df.index:
        sessions = predicted_df.loc[sub]
        ses_times = data.loc[sub]['TIME']
        ses_count = 0
        for session in sessions[0]:
            ses_df = pd.DataFrame(session).rename(columns={0: sub}).T
            ses_df.columns = labels
            ses_df = ses_df.rename_axis('ID')
            # Need to handle separately the subjects with only one visit
            if type(ses_times) == np.float64:
                ses_df.insert(0,'TIME',ses_times)
            else:
                ses_df.insert(0,'TIME',ses_times.iloc[ses_count])
            ses_count += 1 
            df_lst.append(ses_df)

    predicted = pd.concat(df_lst)

    return(predicted)

def predict_timepoints(data, ind, leaspy):
    """
    This function is a kind of reciprocal predict_scores_function. It returns the estimated times at which the
    observed values are reached according to the personalized trajectory.
    """
    # First of all, we need to group all sources into one single feature
    columns = ind.columns[2:]
    labels = leaspy.model.features

    ip_pop = ind[['tau', 'xi']]
    if 'univariate' not in leaspy.model.name:
        ip_pop['sources'] = ''
        for sub in ind.index:
            ip_pop['sources'][sub] = [ind.loc[sub][source] for source in columns]

    mean_time, std_time = leaspy.model.parameters['tau_mean'], max(leaspy.model.parameters['tau_std'], 4)
    timepoints = np.linspace(mean_time - 5 * std_time, mean_time + 8 * std_time, 300)

    #get_age_at_abnormality_conversion(abnormality_thresholds, individual_parameters, timepoints, leaspy)

    df_lst = []
    #predicted_timepoints = 
    for sub in data.index.unique():
        sessions = data.loc[sub]
        ses_count = 0
        for session in sessions.values:
            abnormality_thresholds = {data.columns[i]:session[i] for i in range(1,len(data.columns))}
            individual_parameters = IndividualParameters()
            individual_parameters.add_individual_parameters('index-1', dict(ip_pop.loc[sub]))
            sub_times = pd.DataFrame(get_age_at_abnormality_conversion(abnormality_thresholds, individual_parameters, timepoints, leaspy))
            sub_times['ID'] = sub
            sub_times = sub_times.set_index('ID')
            sub_times.columns = data.columns[1:]
            df_lst.append(sub_times)


    predicted_times = pd.concat(df_lst)
    predicted_times.insert(0, 'TIME', data['TIME'])
    return(predicted_times)
    
def partial_cor_from_precision_matrix(prec_):
    n = len(prec_)
    partial_cor = np.zeros(np.shape(prec_))
    for i in range(n):
        for j in range(n):
            if i == j :
                partial_cor[i][i] = prec_[i][i]
            else:
                partial_cor[i][j] = -prec_[i][j] / np.sqrt(prec_[i][i] * prec_[j][j])
    return(partial_cor)
    
    
def compute_var(data, use_reparametrized=False):
    lst_visits = []
    for sub in data.index.unique():
        data_sub = data.loc[sub]
        if type(data_sub) == pd.core.series.Series:
            continue
        for ses in range(len(data_sub)-1):
            diff = pd.DataFrame(data_sub.iloc[ses+1] - data_sub.iloc[ses]).T
            if use_reparametrized:
                variation = diff / float(diff['reparametrized_TIME'])
                variation['reparametrized_TIME'] = data_sub.iloc[ses]['reparametrized_TIME']    
            else:
                variation = diff / float(diff['TIME'])
                variation['TIME'] = data_sub.iloc[ses]['TIME']
            variation[['bl_'+label for label in list(data.columns[1:5])]] = data_sub.iloc[ses][data_sub.columns[1:5]].values
            lst_visits.append(variation)    
    return(pd.concat(lst_visits))
