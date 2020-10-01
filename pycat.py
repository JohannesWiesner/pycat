"""
Python Adapations to CAT12 functions.
@author: jwiesner
"""

import numpy as np
import pandas as pd

import re

import xml.etree.cElementTree as ET
import os

from nilearn.input_data import NiftiMasker
from scipy.spatial.distance import cdist
from scipy.stats import iqr
from adjustText import adjust_text

import seaborn as sns
import matplotlib.pyplot as plt

def get_TIV(sample,sub_dir_name='anat_1'):
    '''Extract TIV from CAT12 XML Files.
    This function does provide the same functionality as CAT12's third
    module called "Estimate Total Intracranial Volume (TIV)".
    
    Parameters
    ----------
    sample_csv_path: str, or pd.DataFrame
        Path to the sample .csv file, or sample DataFrame
    
    sub_dir_name: str
        Name of the directory for each subject, where CAT12 XML File 
        is stored. The name of the directory has to be specified without
        the path separator.
    
    Returns
    -------
    sample_df: pd.DataFrame
        pandas DataFrame with the added TIV column.
    
    '''

    if isinstance(sample,str):
        sample_df = pd.read_csv(sample)
    else:
        sample_df = sample
        
    sub_dirs = []
    

    # Note: it is assumed that file is nested in a top-directory
    # that again contains a folder called 'report'. For that, go
    # two directories 'up' to get this top-directory.
    for filepath in sample_df['filepath']:
        filepath = os.path.normpath(filepath)
        sub_dir = os.path.dirname(os.path.dirname(filepath))
        sub_dirs.append(sub_dir)
    

    # walk through all subdirectories, search for 'report' folder and get “cat_*.xml 
    # files that contain TIV information
    xml_filepath_list = []
    
    for sub_dir in sub_dirs:
        for i in os.scandir(sub_dir):
            if i.is_dir() and i.path.endswith('report'):
                for (paths, dirs, files) in os.walk(i):
                    for file in files:
                        if file.lower().endswith('.xml'):
                            xml_filepath_list.append(os.path.join(paths, file))
                     
                        
    # extract TIV information from each CAT12 XML file
    tiv_list = []
    
    for xml_path in xml_filepath_list:
        
        root = ET.parse(xml_path).getroot()
        
        for measures in root.findall("./subjectmeasures"):
            for vol in measures.findall(".vol_TIV"):
                vol_tiv = vol.text
                tiv_list.append(vol_tiv)
    
    # add TIV column to sample data frame
    sample_df['tiv'] = tiv_list
    
    return sample_df

# TO-DO: color mark outliers (e.g. 'Patients', 'Controls')
def check_sample_homogeneity(imgs,mask_img,record_ids,average_type='mean',metric='euclidean',fence_factor = 1.5,dst_dir=None,filename=None,memory=None):
    '''Check MRI Image Sample Homogeneity using Distance Measurement.
    
    Parameters
    ----------
    imgs: list, pd.Series
        A list of image paths.
    
    mask_img: 3D Niimg-like object
        A mask image to mask images to brain space.
        
    record_ids: pd.Series
        A series containing unique ids for each subject. The unique ids
        are used for marking outliers in the boxplot.
    
    average_type: str
        How should the average subject be calculated? Choose between 'mean'
        or 'median'.
    
    metric: str or callable
        The metric to use when calculating between instances in a feature
        array. See sklearn.metrics.pairwise.paired_distances.
    
    fence_factor: float
        The factor with which the interquartile range is multiplied with to 
        obtain lower and upper bounds used for identifying outliers.
    
    dst_dir: str or None
        A path pointing to the directory where the plot should be saved.
        If None, the plot is not saved.
    
    filename: str
        The name of the saved file.
    
    memory: instance of joblib.Memory or string
        Used to cache the masking process. By default, no caching is done. 
        If a string is given, it is the path to the caching directory.
    
    Notes
    -----
    See: http://www.neuro.uni-jena.de/vbm/check-sample-homogeneity/
    '''
    if not isinstance(imgs, np.ndarray): 
        niftimasker = NiftiMasker(mask_img=mask_img,
                                  memory=memory)
        imgs_data = niftimasker.fit_transform(imgs)
    
    else: 
        imgs_data = imgs
        
    # calculate the mean data. create an array of sample size with this mean
    # data. 
    if average_type == 'mean':
        avg_img_data = np.mean(imgs_data,axis=0,keepdims=True)
    elif average_type == 'median':
        avg_img_data = np.median(imgs_data,axis=0,keepdims=True)
        
    # calculate the distance from each subject to the average subject.
    distances = cdist(imgs_data,avg_img_data,metric=metric).ravel()

    # get outliers
    interquartile_range = iqr(distances)
    q1 = np.quantile(distances,0.25)
    q3 = np.quantile(distances,0.75)
    lower_bound = q1 - fence_factor * interquartile_range
    upper_bound = q3 + fence_factor * interquartile_range
    
    outlier_indices = np.where((distances < lower_bound) | (distances > upper_bound),True,False)
    outlier_ids = record_ids[outlier_indices]
    outlier_values = distances[outlier_indices]
    
    # boxplot data
    plt.figure()
    boxplot = sns.boxplot(distances)
    boxplot.set_xlabel('Distance to Average')
    sns.despine(left=True)
    
    # annotate boxplot with IDs of outliers
    texts = [plt.text(x=outlier_values[idx],y=0,s=outlier_id,rotation=90) for idx,outlier_id in enumerate(outlier_ids)]
    
    # BUGFIX: When adjusting, text seems to be outside the figure (disable despine to see this effect.)
    # Text should be close to the points. It should be allowed to shift on the x axis but NOT on the y-axis.
    adjust_text(texts,arrowprops=dict(arrowstyle="->", color='black', lw=0.75))
    
    plt.tight_layout()
    
    if dst_dir:
        if not filename:
            raise ValueError('Please provide filename')
    
        dst_path = dst_dir + filename
        plt.savefig(dst_path,dpi=600)

    return outlier_indices