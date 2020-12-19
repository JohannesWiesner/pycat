"""
Python adapations of CAT12's third module (Estimate Total Intracranial
Volume (TIV)) and fourth module (Check Sample).
@author: jwiesner
"""

import numpy as np
import pandas as pd
import os
import xml.etree.cElementTree as ET
from nisupply import get_filepath_df
from nilearn.input_data import NiftiMasker
from scipy.spatial.distance import cdist
from scipy.stats import iqr
from adjustText import adjust_text

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _get_single_report_dir(filepath):
    '''Create a path to a report directory based on a filepath that points
    to a file which was preprocessed using CAT12. This function assumes that 
    each file is placed in a directory called 'mri' that is on the same level 
    as a directory called 'report' (following CAT12s convention for saving the 
    preprocessing output)'''
    
    filepath = os.path.normpath(filepath)
    participant_dir = os.path.dirname(os.path.dirname(filepath))
    report_dir = os.path.join(participant_dir,'report')
    
    if not os.path.isdir(report_dir):
        raise ValueError('This report directory does not exist: {}'.format(report_dir))
    
    return report_dir

def _check_for_same_parent_dir(filepath_df):
    '''Check if all filepaths have the same parent directory'''
    
    filepaths = filepath_df['filepath'].map(os.path.dirname)
    filepaths_array = filepaths.to_numpy() 
    same_parent_dir = (filepaths_array[0] == filepaths_array).all()
    
    if same_parent_dir == False:
        raise ValueError('Not all your files have the same parent directory')

def _get_single_tiv(cat_xml_filepath):
    '''Extract TIV value from one "cat_*.xml" file as created by CAT12'''
    
    root = ET.parse(cat_xml_filepath).getroot()
        
    for measures in root.findall("./subjectmeasures"):
        for vol in measures.findall(".vol_TIV"):
            vol_tiv = vol.text
    
    return vol_tiv
    
def _strip_cat_12_tags(cat_12_filepath):
    '''Take a filepath and return the filename without any tags that are added by CAT12'''
    
    cat_12_tags = ['mwp1','cat_']
    filename = os.path.splitext(os.path.basename(cat_12_filepath))[0]
    
    for tag in cat_12_tags:
        filename = filename.replace(tag,'')
    
    return filename

def add_cat_12_measures(filepath_df,bids_conformity=False):
    '''Add CAT12 information based on an input data frame which contains 
    both participant IDs corresponding filepaths that point to CAT12 preprocessed images. 
    
    Parameters
    ----------
    filepath_df : pd.DataFrame
        A dataframe that contains a column named 'participant_id' and
        a column named 'filepath' as created from nisupply module.
        
    bids_conformity : boolean, optional
        If bids_conformity is False, it is assumed that all preprocessed
        subject images are in the same 'mri' directory. One path for the report 
        directory is created using the first filepath in the dataframe. For this, 
        check that all files have the same parent directory. 
        
        If bids_conformity is True, it is assumed that the 
        preprocessed images are found in a BIDS-conform folder structure.
        
        The default is False.

    Returns
    -------
    filepath_df: pd.DataFrame
        The input dataframe with three added columns (filename,cat_xml_filepath,
        and tiv)

    '''
    
    if bids_conformity == True:
        report_dir = filepath_df['filepath'].map(_get_single_report_dir)
        
    elif bids_conformity == False:
        _check_for_same_parent_dir(filepath_df)
        report_dir = _get_single_report_dir(filepath_df.loc[filepath_df.index[0],'filepath'])
        
    # find "cat_*.xml" files
    cat_xml_filepath_df = get_filepath_df(participant_ids=filepath_df['participant_id'],
                                          src_dir=report_dir,
                                          file_suffix='.xml',
                                          file_prefix='cat',
                                          preceding_dirs='report')
    
    # extract TIV info from each "cat_*.xml" file
    cat_xml_filepath_df['tiv'] = cat_xml_filepath_df['filepath'].map(_get_single_tiv)
    
    # obtain original filenames without tags that where added by CAT12
    # in both the input filepath df and cat_xml_filepath_df
    cat_xml_filepath_df['filename'] = cat_xml_filepath_df['filepath'].map(_strip_cat_12_tags)
    filepath_df['filename'] = filepath_df['filepath'].map(_strip_cat_12_tags)
    
    # merge original filepath_df with cat_xml_filepath_df based on participant_id and original_filename
    filepath_df = pd.merge(filepath_df,cat_xml_filepath_df,on=['participant_id','filename'])
    
    return filepath_df

def check_sample_homogeneity(imgs,mask_img,participant_ids,group_labels=None,idx_annotations=True,
                             average_type='mean',metric='euclidean',fence_factor=1.5,dst_dir=None,
                             filename=None,memory=None):
    '''Check sMRI Image Sample Homogeneity using Distance Measurement. This 
    function provides similar functionality as CAT12's fourth module called 
    "Check Sample".
    
    Parameters
    ----------
    imgs: list, pd.Series
        A list of image paths.
    
    mask_img: 3D Niimg-like object, None
        A mask image to mask images to brain space. If None, default settings
        of nilearn.input_data.NiftiMasker will be used.
        
    participant_ids: pd.Series
        A series containing unique ids for each subject. The unique ids
        are used for marking outliers in the boxplot.
    
    group_labels: pd.Series
        A series containing labels for different groups (e.g. 'patient', 'control')
    
    idx_annotations: Boolean
        If True, annotations will be plotted as row indices. In addition a textbox
        will be plotted that maps each index to the corresponding participant ID.
        
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
    
    memory: instance of joblib.Memory, string or None
        Used to cache the masking process. By default, no caching is done. 
        If a string is given, it is the path to the caching directory.
    
    Notes
    -----
    See: http://www.neuro.uni-jena.de/vbm/check-sample-homogeneity/
    '''
    
    if not isinstance(imgs, np.ndarray): 
        niftimasker = NiftiMasker(mask_img=mask_img,memory=memory)
        imgs_data = niftimasker.fit_transform(imgs)
    else: 
        imgs_data = imgs
        
    # Calculate the average. Create an array of sample size with this average
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
    
    outlier_booleans = np.where((distances < lower_bound) | (distances > upper_bound),True,False)
    outlier_ids = participant_ids[outlier_booleans]
    outlier_values = distances[outlier_booleans]
    outlier_indices = np.where(outlier_booleans)[0]
    
    # boxplot data
    plt.figure()
    flierprops = dict(marker='o',markerfacecolor='0.75',markersize=2.5,linestyle='none')
    boxplot = sns.boxplot(x=distances,whis=fence_factor,flierprops=flierprops)
    boxplot.set_xlabel('Distance to Average')
    
    if idx_annotations:
        # use row indices as annotations
        texts = [plt.text(x=outlier_values[idx],y=0,s=outlier_idx) for idx,outlier_idx in enumerate(outlier_indices)]
        
        # create textbox that maps row indices to participant ids
        textbox_strings = [str(outlier_idx) + ': ' + outlier_id for outlier_idx,outlier_id in zip(outlier_indices.tolist(),outlier_ids.tolist())]
        sep = '\n'
        textbox_content = sep.join(textbox_strings)
        textbox_props = dict(boxstyle='round',facecolor='white',alpha=0.5)
        plt.gcf().text(1.0,1.0,textbox_content,fontsize=8,verticalalignment='top',bbox=textbox_props)
        
    else:
        # use participant ids as annotations
        texts = [plt.text(x=outlier_values[idx],y=0,s=outlier_id) for idx,outlier_id in enumerate(outlier_ids)]
    
    # if group labels are provided, color texts in different colors for each 
    # group and add legend
    if group_labels is not None:
        
        unique_group_labels = group_labels.unique()
        current_palette = sns.color_palette()
        colors = []
        
        for idx in range(0,len(unique_group_labels)):
            colors.append(current_palette[idx])
        
        label_color_dict = dict(zip(unique_group_labels,colors))
        
        outlier_groups = group_labels[outlier_booleans]
    
        for label,text in zip(outlier_groups,texts):
            text.set_color(label_color_dict[label])
        
        patches = [mpatches.Patch(color=color, label=label) for color,label in zip(colors,unique_group_labels)]
        plt.legend(handles=patches)
    
    # use adjustText to properly place annotations
    adjust_text(texts,arrowprops=dict(arrowstyle="-",color='black',lw=0.75))

    plt.tight_layout()
    
    if dst_dir:
        if not filename:
            raise ValueError('Please provide filename')
    
        dst_path = dst_dir + filename
        plt.savefig(dst_path,dpi=600)

    return outlier_indices

if __name__ == '__main__':
    pass