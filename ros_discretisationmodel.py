import math
import numpy as np
from discretisation import *


class DiscModel(object): # Discretisation for non uniform polar discretisation
	def __init__(self,discretisation_dictionary):
		self.bin_info = discretisation_dictionary.values()
		self.state_names = discretisation_dictionary.keys()
		#self.dist_bins_per_angle = [5,5,7,9,9,7,5,5]
		self.get_dims()
	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = []
		for n,k in enumerate(quantity_vector):	
			disc_vector = self.bin_info[n]
			bins.append(discretise(k,disc_vector))
		return map(int,bins)
	def binsToQuantity(self,bins,sample = False):
		quantity = []
		bins = map(int,bins)
		#print bins
		if sample==True:
			for n,k in enumerate(bins):
				if k != (len(self.bin_info[n]) - 1):
					quantity.append(np.random.uniform(self.bin_info[n][k],self.bin_info[n][k+1]))
					#print self.bin_info
				else:
					diff =self.bin_info[n][k] - self.bin_info[n][k-1]  
					quantity.append(np.random.uniform(self.bin_info[n][k], self.bin_info[n][k]+diff))
		else:
			#print sample
			for n,k in enumerate(bins):
				quantity.append(self.bin_info[n][k])
		return quantity
	def binsToState(self,bin_numbers):
		state = 0
		for i in range(0,len(bin_numbers)):
				state+= bin_numbers[i] * np.prod(self.dims[0:i])
		return int(state)
		#Determine where you have ones.
	def stateToBins(self,state):
		#first determine direction.
		bins = np.zeros(len(self.dims))
		for i in range(0,len(self.dims)):
			bins[i] = state%np.prod(self.dims[i])
			state =(math.floor(state/np.prod(self.dims[i])))
		return bins
	def stateToQuantity(self,state):
		bins = self.stateToBins(state)
		return self.binsToQuantity(bins)
	def quantityToState(self,quantity):
		bins = self.quantityToBins(quantity)
		return self.binsToState(bins)
if __name__=="__main__":

	dictionary = ({"robot_x":np.linspace(0,3,15), "robot_y":np.linspace(-math.pi,math.pi,17)[0:16],
					"person_distance":np.linspace(0,4,10),"person_angle": np.linspace(-math.pi,math.pi,9)[0:8]},"person_orentaiton":np.linspace(-math.pi,math.pi,9)[0:8])


	m = DiscModel(dictionary)

	state = m.quantityToState

