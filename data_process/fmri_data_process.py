import nibabel as nib
import numpy as np


def nii_to_np(filename):
    """
    this function is for trans the nii file into numpy array
    :param filename: String
    :return: numpy array
    """
    data = np.asarray(nib.load(filename).get_fdata())
    return data
