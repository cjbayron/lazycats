"""Tests for NumPy utility functions"""
import numpy as np
import src.lazycats.np as catnp
import pytest

def test_top_k_indices_normal():
	arr = [[1,2,3],[4,6,5]]
	k = 2
	exp = np.array([[2,1],[1,2]])
	res = catnp.top_k_indices(arr, k)
	assert(np.all(res==exp))

def test_top_k_indices_normal2():
	arr = [1,2,3,4,6,5]
	k = 2
	exp = np.array([4,5])
	res = catnp.top_k_indices(arr, k)
	assert(np.all(res==exp))

def test_top_k_indices_normal3():
	arr = [
		[[1,2,3],[4,6,5]],
		[[9,8,7],[7,6,11]]
	]
	k = 2
	exp = np.array([
		[[2,1],[1,2]],
		[[0,1],[2,0]]
	])
	res = catnp.top_k_indices(arr, k)
	assert(np.all(res==exp))

def test_top_k_indices_large_k():
	arr = [[1,2,3],[4,6,5]]
	k = 5
	exp = np.array([[2,1,0],[1,2,0]])
	res = catnp.top_k_indices(arr, k)
	assert(np.all(res==exp))

def test_contiguous_lengths_normal():
	arr = [1,1,1,1,1]
	exp = np.array([5])
	res = catnp.contiguous_lengths(arr)
	assert(np.all(res==exp))

def test_contiguous_lengths_normal2():
	arr = [1,1,1,1,1,2,2,3,3,3,3]
	exp = np.array([5,2,4])
	res = catnp.contiguous_lengths(arr)
	assert(np.all(res==exp))

def test_contiguous_lengths_error():
	arr = [[1,1,1,1,1,2,2,3,3,3,3]]
	with pytest.raises(AssertionError):
		catnp.contiguous_lengths(arr)

def test_squash_consecutive_duplicates_normal():
	arr = [1,1,1,1,1,0]
	exp = np.array([1,0])
	res = catnp.squash_consecutive_duplicates(arr)
	assert(np.all(res==exp))

def test_squash_consecutive_duplicates_normal2():
	arr = [1,1,1,1,1,2,2,3,3,3,3]
	exp = np.array([1,2,3])
	res = catnp.squash_consecutive_duplicates(arr)
	assert(np.all(res==exp))

def test_squash_consecutive_duplicates_error():
	arr = [[1,1,1,1,1,2,2,3,3,3,3]]
	with pytest.raises(AssertionError):
		catnp.squash_consecutive_duplicates(arr)

def test_divide_to_subsequences_normal():
	arr = [1,1,1,1,1,2,2,3,3,3,3]
	exp = np.array([[1,1,1,1,1],[2,2,3,3,3],[0,0,0,0,3]])
	res = catnp.divide_to_subsequences(arr, sub_len=5)
	assert(np.all(res==exp))

def test_divide_to_subsequences_normal2():
	arr = [1,1,1,1,1,2,2,3,3,3,3]
	exp = np.array([[1,1,1,1,1],[2,2,3,3,3],[3,4,4,4,4]])
	res = catnp.divide_to_subsequences(arr, sub_len=5, 
			pad=4, pre_pad=False)
	assert(np.all(res==exp))

def test_divide_to_subsequences_normal3():
	arr = [[1,1],[1,1],[1,2],[2,3],[3,3]]
	exp = np.array([[[1,1],[1,1]],
					[[1,2],[2,3]],
					[[0,0],[3,3]]])
	res = catnp.divide_to_subsequences(arr, sub_len=2)
	assert(np.all(res==exp))

def test_divide_to_subsequences_normal4():
	arr = [[1,1],[1,1],[1,2],[2,3],[3,3]]
	exp = np.array([[[1,1],[1,1]],
					[[1,2],[2,3]],
					[[7,4],[3,3]]])
	res = catnp.divide_to_subsequences(arr, pad=[7,4], sub_len=2)
	assert(np.all(res==exp))

def test_divide_to_subsequences_error():
	arr = [[1,1],[1,1],[1,2],[2,3],[3,3]]
	with pytest.raises(AssertionError): # invalid pad override
		catnp.divide_to_subsequences(arr, pad=7, sub_len=2)
