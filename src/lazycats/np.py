"""NumPy utility functions"""
import numpy as np

def top_k_indices(arr, k, axis=1):
    """Get top `k` indices in innermost axis"""
    if axis != 1:
        raise NotImplementedError
    
    temp = np.array(arr) # copy
    top_k_ixs = []
    for _ in range(k): # no checking if k > (array length along axis of argmax)
        max_ixs = np.argmax(temp, axis=axis)
        top_k_ixs.append(max_ixs)
        temp[(range(len(temp)), max_ixs)] = -np.inf # remove from next comparison
    
    top_k_ixs = np.hstack([np.array([ixs]).T for ixs in top_k_ixs])
    return top_k_ixs


def contiguous_lengths(arr):
    """Get lengths of contiguous elements"""
    assert(len(arr.shape) == 1)
    change_points = np.where(arr[1:]-arr[:-1])[0] + 1 # find where values change
    if len(arr) not in change_points:
        change_points = np.append(change_points, len(arr))

    # compute change point relative to previous change point; this essentially computes
    # the length before the value changes
    return np.concatenate(([change_points[0]], change_points[1:]-change_points[:-1]))


def squash_consecutive_duplicates(arr):
    """Squash contiguous sections into single elements"""
    assert(len(arr.shape) == 1)
    # find where values change
    # this is the first index of any consecutive sequence of same values (except element 0)
    change_points = np.where(arr[1:]-arr[:-1])[0] + 1
    return np.concatenate((arr[0:1], arr[change_points]))
