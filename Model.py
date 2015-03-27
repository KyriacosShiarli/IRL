import numpy as np
import math
from discretisationmodel import *
import time
import scipy.spatial as spat
from dataload import extract_info
from kinematics import staticGroupSimple,staticGroupSimple2
from learn_transition import learn_correction,predict_correction
import functions as fn
import cPickle as pickle


class Transition(object):
	def __init__(self):
		self.forward = []
		self.backward = []
		self.tot_states = 0
		self.tot_actions = 0

class Model(object):
	def __init__(self,discretisation,learn  = False):
		self.disc = discretisation#discretisation model

		self.buildFeatureFunction()
		if learn == True:
			self.estimators,ex = learn_correction(10,10)
			print "learned"
		else:
			self.estimators = None
		self.buildTransitionFunction(15)
		self.w = -10*np.ones(len(self.feature_f[0,0,:]))	
		self.reward_f_initial = self.buildRewardFunction()
		self.reward_f = self.buildRewardFunction()
	def convert_to_dense(self,inpt):
		num_states = self.disc.tot_states
		num_actions = self.disc.tot_actions
		al = []
		chunks = []
		for i in range(num_states):
			new_state = True
			for j in range(num_actions):
				values = inpt[j+i*num_actions].values()
				keys = map(int,inpt[j+i*num_actions].keys())
				if keys ==[]:
					if new_state:
						chunks.append([0]);chunks[i].append(chunks[i][j])
						new_state = False
					else:
						chunks[i].append(chunks[i][j])
					print "WARNING"
				if keys !=[]:
					ac = np.ones(len(keys))*j
					col = np.vstack([ac,keys,values])
					if new_state:
						al.append(col)
						new_state = False
						chunks.append([0])
						chunks[i].append(len(keys))
					else:
						al[i] = np.hstack([al[i],col])
						chunks[i].append(len(keys)+chunks[i][j])
		return al,chunks
	def buildTransitionFunction(self,bin_samples):
		self.transition = Transition()
		tot_states=self.transition.tot_states = self.disc.tot_states
		tot_actions=self.transition.tot_actions =self.disc.tot_actions
		self.transition.backward = self.transition.forward= [{} for j in xrange(tot_states*tot_actions)];
		for i in xrange(self.disc.tot_states):
			for j in xrange(self.disc.tot_actions):
				bins = self.disc.stateToBins(i)	
				for k in xrange(bin_samples):
					assert bin_samples != 0
					if bin_samples==1:
						samp = False
					else:
						samp = True
					quantity = self.disc.binsToQuantity(bins,sample = samp)
					action = self.disc.indexToAction(j)
					next_quantity = self.kinematics(quantity,action)
					next_state = self.disc.quantityToState(next_quantity)
					if str(next_state) in self.transition.backward[j + i*self.disc.tot_actions].keys():
						self.transition.backward[j + i*self.disc.tot_actions][str(next_state)] += 1
					else:
						self.transition.backward[j + i*self.disc.tot_actions][str(next_state)] = 1
				v = self.transition.backward[j + i*tot_actions].values()			
				for k in self.transition.backward[j + i*tot_actions].keys():
					self.transition.backward[j + i*tot_actions][k] /= float(np.sum(v))
					self.transition.forward[j + int(k)*tot_actions][str(i)] = self.transition.backward[j+i*tot_actions][k]
				self.transition.dense_backward,self.transition.chunks_backward = convert_to_dense(self.transition_backward)
				self.transition.dense_forward,self.transition.chunks_forward = convert_to_dense(self.transition_forward)
	def buildFeatureFunction(self):
		for i in xrange(self.disc.tot_states):
			state_quantity = self.disc.stateToQuantity(i)
			for j in xrange(self.disc.tot_actions):
				action_quantity = self.disc.indexToAction(j)
				features = self.disc.quantityToFeature(state_quantity,action_quantity) #Still hacky. Need to find a solution to his
				if i == 0 and j == 0: feature_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,len(features)])
				feature_f[j,i,:] = features
				if np.floor(sum(features)+0.1)!=6:
					print "Feature not right"
					print features,state_quantity,action_quantity
	 	self.feature_f = feature_f
	def buildRewardFunction(self):
		return np.dot(self.feature_f,self.w)
	def kinematics(self,state,action):
		if self.estimators == None:
			next_state = self.disc.kinematics(state,action)
		else:
			correction = predict_correction(state,action,self.estimators)
			next_k = state_to_cartesian(self.disc.kinematics(state,action))
			next_state = state_to_polar(next_k - correction)
		return next_state

if __name__ == "__main__":
	w =[-1.,-0.6,-1.8,-1.5,-1.3,-1.2,-1.,-1.,-1.,-1.,-0.7,-1.,-1.,-1,-1,-1.,-1.,-1,-1,-1.,-1.,-0.6,-1,-1]
	d = DiscModel()
	m = Model(d,w,learn = False)
	print m.transition.dense_backward[1][0,:]
	print m.transition.dense_backward[1][1,:]
	#for i in m.feature_f:
		#print i
	#print d.states_per_angle
	#print m.transition_backward
	#print m.transition_forward
	#print rew
	#print m.transition_f[8,1:30,1:30]
	#st = [1,4,0.0,0.0]
	#action = [0.5,0.44]
	#out = m.kinematics(st,action)
	#print out
