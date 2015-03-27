import numpy as np
import math

def discretise(quantity,disc_vector):
	#Returns the bin in which the quantity falls according to the discretisation vector
	typ = len(disc_vector.shape)
	if typ == 1: # 1D discretisation vector of icreasing values
		size = disc_vector.shape[0]
		for i in range(size):
			#assert disc_vector[i]<=disc_vector[i+1] # 1D Discretisation vector should be increasing
			if i == 0 and quantity < disc_vector[i]:
				return i
			elif i != (size-1) and quantity >= disc_vector[i] and quantity < disc_vector[i+1]:
				return i
			elif i == (size - 1):
				return i
	elif typ==2: #2D discretisation to be used for angles
		for i,j in enumerate(disc_vector):
			if quantity >= j[0] and quantity < j[1]:
				return i
def discretise_with_overlap(quantity,disc_vector):
	#Returns the bin/s in which the quantity falls according to the discretisation vector which can overlap
	#if overlap returns a vector. If there is no ovelap returns a number. To be used for tile coding when producing features.
	#DISC VECTOR NEEDS TO BE A NUMPY ARRAY.
	out = np.zeros(len(disc_vector))
	size = disc_vector.shape[0]
	for i in range(len(disc_vector)):
		assert disc_vector[i,0]<=disc_vector[i,1], "%s"%disc_vector[i,0] # 1D Discretisation vector should be increasing
		if quantity >= disc_vector[i,0] and quantity < disc_vector[i,1]:
			out[i] = 1
		if i == (size - 1) and quantity >= disc_vector[i,1]:
			out[i]=1
	return out


def discretise_with_overlap(quantity,disc_vector):
	#Returns the bin/s in which the quantity falls according to the discretisation vector which can overlap
	#if overlap returns a vector. If there is no ovelap returns a number. To be used for tile coding when producing features.
	#DISC VECTOR NEEDS TO BE A NUMPY ARRAY.
	out = np.zeros(len(disc_vector))
	size = disc_vector.shape[0]
	for i in range(len(disc_vector)):
 # 1D Discretisation vector should be increasing
		if i == 0 and quantity < disc_vector[0]:
			out[i] = 1
		if i<(size-1) and quantity >= disc_vector[i] and quantity < disc_vector[i+1]:
			assert disc_vector[i]<=disc_vector[i+1], "%s"%disc_vector[i]
			out[i] = 1
		if i == (size - 1) and quantity >= disc_vector[i]:
			out[i]=1
	return out

def discretise2(quantity,disc_vector):
	#Returns the bin in which the quantity falls according to the discretisation vector
	out = np.zeros(len(disc_vector))	
	size = disc_vector.shape[0]
	for i in range(size):
		#assert disc_vector[i]<=disc_vector[i+1] # 1D Discretisation vector should be increasing
		if i == 0 and quantity < disc_vector[i]:
			out[i] = 1
			return out
		elif i != (size-1) and quantity >= disc_vector[i] and quantity < disc_vector[i+1]:
			out[i] = 1
			return out
		elif i == (size - 1):
			out[i] = 1
			return out
if __name__ == "__main__":
	x = np.linspace(1,4.5,9)
	y = 4.5
	z = normalised_gaussian_response(y,x)
	print sum(z)

