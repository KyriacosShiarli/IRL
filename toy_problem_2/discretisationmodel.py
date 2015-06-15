import math
import numpy as np
#from RFeatures import *
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
	def __init__(self,kinematics = staticGroup_with_target,actions = {"linear" :np.array([0,1,2,3,4])},feature =  feature_full):
		obstacle_x = np.linspace(0,15,1) # Nine bins whatever the case
		obstacle_y = np.linspace(0,15,1) # Nine bins whatever the case
		obstacle_v = np.array([0,1,2,3,4])
		target_x = np.linspace(0,15,1) # Target orientation bsins
		target_y = np.linspace(0,15,1)
		self.feature = feature
		self.actions = actions#these are in the form of a dictionary. but they really should be
		self.bin_info = np.array([obstacle_x,obstacle_y,obstacle_v,target_x,target_y])
		self.kinematics = kinematics #change kinematics
		self.get_dims()
		self.tot_states =sum(self.states_per_angle)
		self.tot_actions = len(self.actions["linear"])*len(self.actions["angular"])
		print self.tot_states
	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = np.zeros(self.bin_info.shape)
		for n,k in enumerate(quantity_vector):
			temp = np.zeros( len(self.bin_info[n]))
			temp[k] = 1
			bins.append(temp)
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
			if i !=len(bin_numbers)-1:
				state+= bin_numbers[i] * np.prod(self.dims[i+1::])
			else:
				state+=bin_numbers[i]
		return state
		#Determine where you have ones.
	def stateToBins(self,state):
		#first determine direction.
		bins = []
		for i in range(0,len(self.dims)):
			bins.append(math.floor(state/np.prod(self.dims[i::])))
			state = state%np.prod(self.dims[i::])
		bins.append(state)
		return bins
	def stateToQuantity(self,state):
		bins = self.stateToBins(state)
		return self.binsToQuantity(bins)
	def quantityToState(self,quantity):
		bins = self.quantityToBins(quantity)
		return self.binsToState(bins)
	def actionToIndex(self,action):
		return action
	def indexToAction(self,index):
		return index
	def quantityToFeature(self,state,action=None):
		state = self.stateToQuantity(self.quantityToState(state))
		feature = self.feature(state,action)
		return feature
if __name__=="__main__":
		m = DiscModel()
		#m.indexToAction(0)
		quantity = [0,0,2,3,4]
		state = m.quantityToState(quantity)
		print state

	#m.build_discretisation_map()

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

	#s = [-0.39269908169872414, 1.125, 2.3561944901923448, 1.3333333333333333]
	#a =  [-0.20000000000000007, 0.20000000000000001]
	#print m.quantityToFeature(s,a)