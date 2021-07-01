"""NumPy utility functions"""
import numpy as np

def top_k_indices(arr, k):
    """Get top `k` indices in INNERMOST axis"""
    temp = np.array(arr).astype(float) # copy
    if k > temp.shape[-1]:
        k = temp.shape[-1]

    top_ixs = np.flip(np.argsort(arr, axis=-1), axis=-1)
    top_k_ixs = np.take(top_ixs, range(k), axis=-1)
    return top_k_ixs


def contiguous_lengths(arr):
    """Get lengths of contiguous elements"""
    arr = np.array(arr)
    assert(len(arr.shape) == 1)
    change_points = np.where(arr[1:]-arr[:-1])[0] + 1 # find where values change
    if len(arr) not in change_points:
        change_points = np.append(change_points, len(arr))

    # compute change point relative to previous change point; this essentially computes
    # the length before the value changes
    return np.concatenate(([change_points[0]], change_points[1:]-change_points[:-1]))


def squash_consecutive_duplicates(arr):
    """Squash contiguous sections into single elements"""
    arr = np.array(arr)
    assert(len(arr.shape) == 1)
    # find where values change
    # this is the first index of any consecutive sequence of same values (except element 0)
    change_points = np.where(arr[1:]-arr[:-1])[0] + 1
    return np.concatenate((arr[0:1], arr[change_points]))
