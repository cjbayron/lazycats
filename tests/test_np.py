"""Tests for NumPy utility functions"""
import numpy as np
import src.lazycats.np as catnp

def test_top_k_indices_normal():
	arr = [[1,2,3],[4,6,5]]
	k = 2
	exp = np.array([[2,1],[1,2]])
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