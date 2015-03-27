from Discretisation import *
from forwardBackward import *
from Model import Model
from dataload import *
import matplotlib.pyplot as plt
import pp
import numpy as np
import time
from itertools import compress
import matplotlib.pyplot as plt
def forward_par(action_probs,example,model):
	import numpy as np
	import forwardBackward as fb	
	state_frequencies= fb.smart_forward(action_probs,model.transition,example["start_state"],example["steps"])
	cumulative = 0
	likelihood = 0
	log_policy = np.log(action_probs)
	# Gradient Computation - - - - - - - - - - - 
	for idx,i in enumerate(state_frequencies):
		cumulative += i*model.feature_f[idx]
	diff =(example["feature_sum"]-cumulative)
	#feature_diff_avg +=(example["feature_sum"]-cumulative)/example["steps"]
	# Action matching computation Should be adjusted for path length
	for n in range(example["steps"]-1):
		likelihood += max(log_policy[example["action_numbers"][n+1],example["state_numbers"][n]], 
						log_policy[example["action_numbers"][n],example["state_numbers"][n]])
	return diff,likelihood

def extractStatistics_parallel(examples,model,z_states = None,threads=2):
	job_server = pp.Server(int(threads))
	print len(examples)
	batches = np.arange(0,len(examples),threads)
	total_diff = 0
	total_likelihood = 0
	goals = []
	for example in examples:
		goals.append(example["end_state"])
	action_probs,z_states =caus_ent_backward(model.transition,model.reward_f,goals,example["steps"],0.1,z_states)
	#es = examples[1]["state_numbers"][50]
	#ac = examples[1]["action_numbers"][50]
	#plt.scatter(range(len(action_probs[:,es])),action_probs[:,es])
	#plt.scatter(ac,action_probs[ac,es],color = "red")
	#plt.show()
	for i in xrange(len(batches)):
		if i + 1 < len(batches):expls = examples[i:i+1]
		else: expls = examples[i:len(examples)]
		jobs = [job_server.submit(forward_par,args = (action_probs,ex,mod,)) for ex in expls]
		for job in jobs:
			total_diff += job()[0]
			total_likelihood += job()[1] 
	job_server.destroy()
	print "Sum likelihoods",np.sum(total_likelihood/len(examples))
	return total_diff/len(examples), total_likelihood/len(examples),z_states

def extractStatistics(examples,model):
	#performs forward backwward calculation for a reward function and extracts different kinds of statistics used to
	#evaluate the algorithm
	z_states = None
	feature_diff = 0
	feature_diff_avg=0
	goals = []
	likelihoods = np.zeros(len(examples))
	for num,example in enumerate(examples):
		print "In the loop"
		start = time.time()
		#action_probs,z_states =caus_ent_backward(model.transition_backward,model.transition_f,model.reward_f,example["end_state"],example["steps"],200,z_states)
		action_probs,z_states =backward_parallel(model.transition,model.reward_f,example["end_state"],example["steps"],3,200,None)
		state_frequencies= smart_forward(action_probs,model.transition,example["start_state"],example["steps"])
		cumulative = 0
		log_policy = np.log(action_probs)
		# Gradient Computation - - - - - - - - - - - 
		for idx,i in enumerate(state_frequencies):
			cumulative += i*model.feature_f[idx]
		feature_diff +=example["feature_sum"]-cumulative
		#feature_diff_avg +=(example["feature_sum"]-cumulative)/example["steps"]
		# Action matching computation Should be adjusted for path length
		for n in xrange(example["steps"]-1):
			likelihoods[num] += max(log_policy[example["action_numbers"][n+1],example["state_numbers"][n]], 
									log_policy[example["action_numbers"][n],example["state_numbers"][n]])
	feature_diff/=len(examples)
	return feature_diff,np.sum(likelihoods)
def learn(mod,test,train,iterations,gamma):
	grad_array= []
	valid_array = []
	likelihoods_train = []
	likelihoods_test = []
	for iteration in xrange(iterations):
		grad_train,likelihood_train,z_states = extractStatistics_parallel(train,mod)
		grad_array.append(np.sum(np.absolute(grad_train)))
		likelihoods_train.append(likelihood_train)
		grad_test,likelihood_test,z_states = extractStatistics_parallel(test,mod,z_states)
		valid_array.append(np.sum(np.absolute(grad_test)))
		likelihoods_test.append(likelihood_test)
		print "Old W", mod.w 	
		print "Gradient", grad_train
		print "Cross Validation Difference", grad_test
		mod.w=mod.w *np.exp(-gamma*grad_train)
		print "Train Likelihood"
		print "Test Likelihood"
		print "new W" , mod.w
		mod.reward_f = mod.buildRewardFunction()
		print "THIS IS N",iteration
	return grad_array,valid_array,likelihoods_train,likelihoods_test

