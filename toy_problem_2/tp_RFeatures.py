import numpy as np
import math
#from discretisation import *
from scipy.stats import norm
def staticGroupSimple1(distance,orientation_group,max_distance=7):
	features = np.zeros(2)
	features[0] = math.exp(-(3-distance)**2)	
	features[1] = 0.5*(math.cos(orientation_group)+1)
	return features

def toy_problem_features_simple(state,target,boundaries):
	# [xtarget,ytarget,xobs,yobs]
	#xdistance from target
	xtar = np.zeros(boundaries[0]+1)
	dist = abs(state[0]-target[0])
	#print "DIST1",dist
	if dist <=boundaries[0]: 
		xtar[dist]=1
	else:
		xtar[-1]=1
	#ydistance from target
	ytar = np.zeros(boundaries[1]+1)
	dist = abs(state[1]-target[1])
	#print "DIST2",dist
	if dist <=boundaries[1]: 
		ytar[dist]=1
	else:
		ytar[-1]=1
	#xdistance from obstacle
	xobs = np.zeros(boundaries[0]+1)
	dist = abs(state[0]-state[2])
	#print "DIST3",dist
	if dist <=boundaries[0]: 
		xobs[dist]=1
	else:
		xobs[-1]=1
	#ydistance from obstacle
	yobs = np.zeros(boundaries[1]+1)
	dist = abs(state[1]-state[3])
	#print "DIST4",dist
	if dist <=boundaries[1]: 
		yobs[dist]=1
	else:
		yobs[-1]=1

	return np.concatenate([xtar,ytar,xobs,yobs])

def builder_features(state):
	feature = np.zeros(5)
	if state[0]==1:
		feature[0]=1
	#else:
	#	feature[5]=1
	if state[1]==1:
		feature[1]=1
	#else:
	#	feature[6]=1
	if state[4]==1:
		feature[2]=1
	#else:
	#	feature[7]=1
	if state[5]==1:
		feature[3]=1
	#else:
	#	feature[8]=1
	if state[8]==1:
		feature[4]=1
	#else:
	#	feature[9]=1
	return feature



if __name__ == "__main__":
	state = [0,0,5,5,4]
	target = [3,3]
	boundaries = [15,15]
	out = toy_problem_features_simple(state,target,boundaries)
	print out
