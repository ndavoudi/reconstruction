

#import scipy.io as sio
import h5py #load mat file
import os
import numpy as np


def get_data_paths_list(image_folder, mask_folder):
    """Returns lists of paths to each image and mask."""

    image_paths = [os.path.join(image_folder, x) for x in os.listdir(
        image_folder) if x.endswith(".mat")]

    mask_paths = [os.path.join(mask_folder, x) for x in os.listdir(
        mask_folder) if x.endswith(".mat")]

    return image_paths, mask_paths




def _parse_data(image_paths, mask_paths):
    file_image = h5py.File(image_paths,'r') # simulated 3D
    variables_image = file_image.items()
    for var in variables_image:
        name_image = var[0] # image_th
        data_image = var[1]
        if type(data_image) is h5py.Dataset:
            images = data_image.value # NumPy ndArray / Value  #dataset.value has been deprecated. Use dataset[()] instead.


    file_mask = h5py.File(mask_paths,'r') # simulated 3D
    variables_mask = file_mask.items()
    for var in variables_mask:
        name_mask = var[0] # image_th
        data_mask = var[1]
        if type(data_mask) is h5py.Dataset:
            masks = data_mask.value # NumPy ndArray / Value

    images = np.expand_dims(images, axis=3)
    masks = np.expand_dims(masks, axis=3)

    return images, masks
