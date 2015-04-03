import math
import numpy as np
from discretisationmodel import *
from RFeatures import *
from kinematics import *
from scipy.stats import norm
import functions as fn
from fg_gradient import adaboost_reg2

class featureModel(object):
	def __init__(self,function,state_stuff,action_stuff,distributions=None):
		self.function = function
		self.state_stuff = state_stuff
		self.action_stuff = action_stuff
		self.distributions = distributions
	def __call__(self,state,action):
		return self.function(self.state_stuff,self.action_stuff,state,action)
variances = [0.3,0.3,0.3,0.3,0.03,0.03]
distr = [ norm(loc = 0, scale = variances[i]) for i in range(6)]

state_disc_full = [np.linspace(-math.pi,math.pi,9)[0:8],np.linspace(0,3,15),np.linspace(-math.pi,math.pi,17)[0:16],np.linspace(0,4,10)]
action_disc_full = [np.arange(-0.5,0.5,0.1),np.array([0,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6])]
feature_full = featureModel(tile_code_features,state_disc_full,action_disc_full)

state_disc_sym = [np.cos(np.linspace(-math.pi,0,9)[0:8]/2),np.linspace(0,3,9)[[1,3,5,7,8]],np.cos(np.linspace(-math.pi,0,5)[0:4]/2),np.linspace(0,4,4)]
action_disc_sym = [np.arange(-0.5,0.5,0.1),np.array([0,0.1,0.15,0.2,0.25,0.3])]
feature_sym = featureModel(symmetric_angles,state_disc_sym,action_disc_sym)
class DiscModel(object): # Discretisation for non uniform polar discretisation
	def __init__(self,kinematics = staticGroup_with_target,actions = {"linear" :np.array([0,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6]),"angular" : np.arange(-0.5,0.5,0.1)},feature =  feature_full):
		distance = np.linspace(0,3,15) # Nine bins whatever the case
		self.dist_bins_per_angle = [15]*16
		target_angle = np.linspace(-math.pi,math.pi,17)[0:16] # Target orientation bsins
		target_distance = np.linspace(0,4,10)
		angle = np.linspace(-math.pi,math.pi,8)[0:9] # Angle to persons bins
		self.feature = feature
		self.actions = actions#these are in the form of a dictionary
		self.bin_info = [angle,distance,target_angle,target_distance]
		self.discretisation_map = pickle_loader("saved_structures/discretisation_map.pkl")
		
		#self.dist_bins_per_angle = [5,5,7,9,9,7,5,5]
		self.kinematics = kinematics
		self.get_dims()
		self.statesPerAngle()
		self.tot_states =sum(self.states_per_angle)
		self.tot_actions = len(self.actions["linear"])*len(self.actions["angular"])
		print self.tot_states
	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
	def statesPerAngle(self):
		#calculates number of possible states after an orientation is known for all orientiations and returns
		# a vector. This is a direct result of the fact that we have variable length discretisations at different
		# orientations in order to save s
		self.states_per_angle = [] # number of states for each orientation
		for i in range(self.dims[0]):
			self.states_per_angle.append(int(self.dist_bins_per_angle[i] * np.prod(self.dims[2::])))

	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = []
		for n,k in enumerate(quantity_vector):
			if n == 1: 
				disc_vector = self.bin_info[n][:self.dist_bins_per_angle[bins[0]]]
			else: disc_vector = self.bin_info[n]
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
		state = sum(self.states_per_angle[0:int(bin_numbers[0])])
		if bin_numbers[1]>self.dist_bins_per_angle[bin_numbers[0]]-1: print "Illegal state based on model";return None
		for i in range(1,len(bin_numbers)):
			if i !=len(bin_numbers)-1:
				state+= bin_numbers[i] * np.prod(self.dims[i+1::])
			else:
				state+=bin_numbers[i]
		return state
		#Determine where you have ones.
	def stateToBins(self,state):
		#first determine direction.
		bins = []
		for i,dim in enumerate(self.states_per_angle):
			state -=  dim 
			if state < 0:
				state = state + dim
				bins.append(i)
				break
		for i in range(2,len(self.dims)):
			bins.append(math.floor(state/np.prod(self.dims[i::])))
			state = state%np.prod(self.dims[i::])
		bins.append(state)
		if bins[1]>self.dist_bins_per_angle[bins[0]] - 1:
			print "Mistake in calculation. REview Function"
			print bins
			return None
		else:
			return bins
	def stateToQuantity(self,state):
		bins = self.stateToBins(state)
		return self.binsToQuantity(bins)
	def quantityToState(self,quantity):
		bins = self.quantityToBins(quantity)
		return self.binsToState(bins)
	def actionToIndex(self,action):
		linear_idx = discretise(action[1],self.actions["linear"])
		angular_idx = discretise(action[0],self.actions["angular"])
		return linear_idx + angular_idx*len(self.actions["linear"])
	def indexToAction(self,index):
		angular_idx = int(math.floor(index/len(self.actions["linear"])))
		linear_idx = int(index%len(self.actions["linear"]))
		return [self.actions["angular"][angular_idx],self.actions["linear"][linear_idx]]
	def quantityToFeature(self,state,action=None):
		state = self.stateToQuantity(self.quantityToState(state))
		feature = self.feature(state,action)
		return feature
	def build_discretisation_map(self):
		#a discretisation map is a map from state action numbers to states and actions in continouous space.
		#note that the states are going to be saved in cartesian coordinates so that they are not discontinouous
		#this will be mainly used in the functional gradient descent method for states
		all_states = []
		all_actions = []
		discretisation_map =[]
		for i in range(0,self.tot_states):
			state = self.stateToQuantity(i)
			state = fn.state_to_cartesian(state)
			all_states.append(state)
		for j in range(0,self.tot_actions):
			action = self.indexToAction(j)
			all_actions.append(action) 
		for i in all_states:
			for j in all_actions:
				discretisation_map.append(np.hstack([i,j]))
		fn.pickle_saver(discretisation_map,"discretisation_map.pkl")
if __name__=="__main__":
	m = DiscModel()
	m.indexToAction(0)
	#m.build_discretisation_map()

	f = fn.pickle_loader("discretisation_map.pkl")
	f = np.array(f)
	x = f[0:200,:]
	y = np.zeros(m.tot_actions*m.tot_states)
	y = np.random.normal(0,2,len(x))
	reg = adaboost_reg2(x,y,200,5)
	#print m.feature.state_stuff
	#t1 = np.linspace(-math.pi,math.pi,200)
	#t2 = np.linspace(-math.pi,math.pi,200)
	#print m.feature.state_stuff
	#for i in range(200):
	#	state = np.array([t1[i],2,t2[i],2])
	#	action = np.array([0.1,0.2])
		#print "before entering",np.cos(state[0]/2)
	#	feature = m.quantityToFeature(state,action)
	#	print "feature", feature
		#print feature[17:21]

	s = [-0.39269908169872414, 1.125, 2.3561944901923448, 1.3333333333333333]
	a =  [-0.20000000000000007, 0.20000000000000001]
	print m.quantityToFeature(s,a)