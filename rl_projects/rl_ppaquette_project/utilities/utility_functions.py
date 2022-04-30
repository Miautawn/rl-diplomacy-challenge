""" Utility functions
    - Functions for shaping/modifying data or structures
"""

import zlib
import json

import numpy as np
import torch
import torch.nn.functional as F


def pad_tensor(tensor: torch.tensor, axis: int, min_size: int, pad_value = 0):
    """pads the tensor axis with min_size ammount of pad_value"""
    
    new_axis = axis if axis >= 0 else len(tensor.shape) + axis
    
    assert new_axis >= 0 and new_axis <= len(tensor.shape) - 1, \
    f"Tensor with shape {tensor.shape} got invalid shape index {axis}"

    # check whether the tensor axis is already >= min_size
    if tensor.shape[axis] >= min_size:
        return tensor
    
    pad_structure = [0,0]*(len(tensor.shape) - new_axis)
        
    pad_structure[-1] = min_size - tensor.shape[new_axis]
    return F.pad(tensor, tuple(pad_structure), value = pad_value)

def compress_dict(dict_object):
    """
        compresses a dict into a byte string encoded with zlib.
        Numpy arrays will be transformed to python lists.
    """
    
    # convert each np.ndarray to list within the dict    
    def deep_search(converted_dict):     
        for key, value in converted_dict.items():
            if isinstance(value, dict):
                compress_dict(value)
            if isinstance(value, np.ndarray):
                converted_dict[key] = value.tolist()

        return converted_dict

    dict_object = deep_search(dict_object)
    
    #returns encoded game bytes as a string
    encoded_dict = zlib.compress(json.dumps(dict_object).encode("utf-8"), level=-1)
    return str(encoded_dict)   


def decompress_dict(dict_bytes_string, numpy_arr_members = []):
    """
        Turns the zlib encoded dictionary byte string back to dictionary.
        Convertable columumns arguments holds the names of dict memebers that can
        be converted 
    """
    
    #turn from string-bytes to actual bytes
    dict_bytes = eval(dict_bytes_string)

    dict_object = json.loads(zlib.decompress(dict_bytes).decode("utf-8"))
    
    # convert each list within the dict to np.array if it's in numpy_arr_members  
    def deep_search(converted_dict):
        for key, value in converted_dict.items():
            if isinstance(value, dict):
                deep_search(value)
            if isinstance(value, list) and key in numpy_arr_members:
                converted_dict[key] = np.array(value)

        return dict_object
    
    dict_object = deep_search(dict_object)
    
    return dict_object


