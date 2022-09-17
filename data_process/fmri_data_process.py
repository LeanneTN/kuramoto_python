import nibabel as nib
import numpy as np
import pandas as pd


def nii_to_np(filename):
    """
    this function is for trans the nii file into numpy array
    :param filename: String
    :return: numpy array
    """
    data = np.asarray(nib.load(filename).get_fdata())
    return data


def csv_to_np(filename):
    """
    this function is for trans the csv file into numpy array
    :param filename: String
    :return: numpy array
    """
    data = pd.read_csv(filename)
    data = np.array(data)
    return data
