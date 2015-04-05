from discretisationmodel import *
from forwardBackward import *
from Model import Model,Transition
from dataload import *
import numpy as np
from plott import *
from functions import *
import scipy.optimize as opt
from fg_gradient import adaboost_reg2
import os
from evaluation import evaluate_policy
from data_structures import Results,EmptyObject

def replace_with_kinematics(model,examples):
	#Replaces data with data under kinematics. Uses the actions and the initial states of the examples to generate
	#new data under the kinematics so that the problem of having different transition functions is mitigated 
	for example in examples:
		for i in range(1,example.states.shape[0]):
			example.states[i,:] = model.kinematics(example.states[i-1,:],example.actions[i,:])

	return examples
class Learner(object):
	def __init__(self,model,train_good,test_good,train_bad = None,test_bad = None):
		self.model  = model
		self.train_good  = train_good
		self.train_bad  = train_bad
		self.test_good  = test_good
		self.test_bad  = test_bad
		self.policy = None
		self.z_s = None
	def __call__(self,iterations,rate,moment,examples_type = "both",processing = "batch"):

		self.upper_disc_bound = get_model_based_statistics(self.model,self.train_good,1000)
		self.policy_initial,log_policy,z_states = self.inference(self.train_good,None)
		self.policy = self.policy_initial
		self.statistics(self.train_good,self.policy_initial,log_policy)
		w0 =-10*np.ones([model.disc.tot_actions,model.disc.tot_states]) + np.random.normal(0,1,[model.disc.tot_actions,model.disc.tot_states])
		bound = [[-300,-0.001]]*w0.size
		#results = opt.minimize(self.lbfgs,w0,jac = True,method = "L-BFGS-B",bounds = bound)
		results = self.learn(iterations,gamma,moment,examples_type)
	def statistics(self,examples,policy,log_policy):
		steps = examples[0].steps
		start_states = [(te.start) for te in examples]
		feature_avg = 0
		likelihood = 0
		entropy = 0
		for example in examples:
			feature_avg +=example.state_action_counts
			for step in xrange(example.steps-1):
				entropy+=-1 * max(log_policy[example.action_numbers[step+1],example.state_numbers[step]], 
					log_policy[example.action_numbers[step],example.state_numbers[step]])* max(policy[example.action_numbers[step+1],example.state_numbers[step]], 
					log_policy[example.action_numbers[step],example.state_numbers[step]])
				likelihood+=-1 * max(log_policy[example.action_numbers[step+1],example.state_numbers[step]], 
					log_policy[example.action_numbers[step],example.state_numbers[step]])

		state_frequencies,sa= forward(policy,self.model.transition,start_states,steps)

		a,s,f = self.model.feature_f.shape
		#gradient = (feature_avg/len(examples) - 
		#			np.dot(sa.reshape(s*a),self.model.feature_f.reshape(s*a,f)))
		gradient = (np.dot(self.upper_disc_bound.reshape(s*a),self.model.feature_f.reshape(s*a,f)) - 
					np.dot(sa.reshape(s*a),self.model.feature_f.reshape(s*a,f)))
		#gradient = feature_avg/len(examples) - sa

		# Regression fit gradient
		#y = np.array(feature_avg/len(examples)).reshape(s*a) - sa.reshape(s*a)
		#reg = adaboost_reg2(self.model.disc.discretisation_map,y,400,5)
		#gradient = reg.predict(self.model.disc.discretisation_map)
		#gradient = gradient.reshape([a,s])
		#gradient = self.upper_disc_bound - sa
		error = np.sum(np.absolute(self.upper_disc_bound - sa))
		likelihood/=len(examples)
		res = evaluate_policy(self.train_good[:5],self.model,policy)
		print "METRIC", res
		print "LIKELIHOOD=",likelihood 
		entropy/=len(examples)
		print "OBJECTIVE", -entropy - np.sum(np.sum(self.model.w * gradient))
		return gradient,error,entropy
	def inference(self,examples,z_states=None):
		print "GOTHERE"
		steps = examples[0].steps
		goals = [(tr.goal) for tr in examples]
		return caus_ent_backward(self.model.transition,self.model.reward_f,goals,steps,conv = 0.1,z_states=z_states)
	def learn(self,iterations,rate,moment,examples_type):
		z_states = None
		if examples_type == "good" or examples_type == "bad":
			results = Results(iterations)
			for i in xrange(iterations):
				if examples_type == "good":
					assert self.train_good !=None ; assert self.test_good !=None
					policy,log_policy,z_states = self.inference(self.train_good,z_states)
					if i ==0:
						self.policy_initial = policy
					gradient,results.train_error[i],results.train_lik[i] = self.statistics(self.train_good,policy,log_policy)
					cv_diff,results.test_error[i],results.test_lik[i] = self.statistics(self.test_good,policy,log_policy)
				elif examples_type == "bad":
					assert self.train_bad !=None ; assert self.test_bad !=None
					policy,log_policy,z_states = self.inference(self.train_bad,z_states)
					gradient,results.train_error[i],results.train_lik[i] = self.statistics(self.train_bad,policy,log_policy)
					cv_diff,results.test_error[i],results.test_lik[i] = self.statistics(self.test_bad,policy,log_policy)
					rate *= -1
				print "---------THIS IS N: %s---------"%i
				print "Examples are:", examples_type
				print "Train Difference: ", gradient
				print "Train Likelihood: ", results.train_lik[i]
				print "Test Likelihood: ", results.test_lik[i]
				if i != 0: gradient = momentum(gradient,prev,moment)
				self.model.w=self.model.w*np.exp(-rate*gradient) ; print "new W: " , self.model.w
				self.model.reward_f = self.model.buildRewardFunction()
				#self.model.reward_f = self.model.reward_f - 3*self.model.external_reward 
				prev = gradient
				if i < 25:rate = rate/1.01
				else: rate = rate*1.02
				#self.model.reward_f *=np.exp(-rate*gradient)
			self.results = results
			learner.policy = policy
		if examples_type =="both":
			#Initialise data structures
			results_g = Results(iterations)
			results_b = Results(iterations)
			#Make sure the data is there
			assert self.train_good !=None ; assert self.test_good !=None 
			assert self.train_bad !=None ; assert self.test_bad !=None
			#iterate
			for i in xrange(iterations):
				policy,log_policy,z_states = self.inference(self.train_good,z_states)
				if i ==0:
					self.policy_initial = policy
				gradient_g,results_g.train_error[i],results_g.train_lik[i] = self.statistics(self.train_good,policy,log_policy)
				cv_diff,results_g.test_error[i],results_g.test_lik[i] = self.statistics(self.test_good,policy,log_policy)
				#policy,log_policy,z_states = self.inference(self.train_bad,z_states)
				gradient_b,results_b.train_error[i],results_b.train_lik[i] = self.statistics(self.train_bad,policy,log_policy)
				cv_diff,results_b.test_error[i],results_b.test_lik[i] = self.statistics(self.test_bad,policy,log_policy)
				gradient = gradient_g - 0.4*gradient_b
				print "---------THIS IS N:---------",i
				print "Examples are :", examples_type
				print "Train Difference: ", gradient_g
				print "Train Likelihood: ", results_g.train_lik[i]
				print "Test Likelihood: ", results_g.test_lik[i]
				if i != 0: gradient = momentum(gradient,prev,moment)
				#self.model.w=self.model.w*np.exp(-rate*gradient) ; print "new W: " , self.model.w
				self.model.reward_f *=np.exp(-rate*gradient)
				prev = gradient
				if i < 15:rate = rate/1.05
				else: rate = rate*1.05
				#self.model.reward_f = self.model.buildRewardFunction()
			self.results_g = results_g
			self.results_b = results_b
			self.policy = policy
	def lbfgs(self,w0):
		steps = 50
		wr = np.reshape(w0,[self.model.disc.tot_actions,self.model.disc.tot_states])
		print wr
		print "MAXMIN",np.amax(wr),np.amin(wr)
		self.model.reward_f = wr
		print w0.shape
		p,lp,self.z_s = self.inference(self.train_good,self.z_s)
		self.policy = p
		print np.sum(np.sum(self.policy-self.policy_initial))
		grad,error,entropy = self.statistics(self.train_good,p,lp)
		obj = -entropy -0.5*np.dot(np.reshape(self.model.reward_f,self.model.disc.tot_actions*self.model.disc.tot_states) , np.reshape(grad,self.model.disc.tot_actions*self.model.disc.tot_states))
		grad = -1*np.reshape(grad,self.model.disc.tot_actions*self.model.disc.tot_states)
		print "OBJ",obj
		print "ENTROPY",entropy
		print "ERROR",error
		return obj,grad

if __name__ == "__main__":
  	#Initialisa models
  	base_directory = "results/stateonlynormal/"
  	#plot_directory = base_directory + "plots/"
  	#metrics_directory = base_directory + "numbers/"
  	make_dir(base_directory) 

  	disc_model = DiscModel()

  	#model = fn.pickle_loader("saved_structures/model.pkl")
  	dd= 0
  	for i in disc_model.bin_info:
  		dd+=len(i)
  	
  	print "Model"
  	supervised_reward = fn.pickle_loader("saved_structures/supervised_reward_state2.pkl")
  	
  	examples = load_all(30)
  	examples = extract_info(disc_model,"Full",examples)

  	examples_good = []
  	examples_bad = []
  	for i in examples:
  		if i.quality == 1 and i.feature_sum[16]!=30:
  			examples_good.append(i)
  		else:
  			examples_bad.append(i)

	#examples_good = fn.pickle_loader("saved_structures/examples_good.pkl")
	#examples_bad = fn.pickle_loader("saved_structures/examples_bad.pkl")
	#model = pickle_loader("saved_structures/new_model.pkl")
	#model.disc = disc_model
	model = Model(disc_model,learn = True)
	print "Feature function"
	print "end Feature function"
	model.feature_f[:,:,dd:] = 0
	pickle_saver(model,"saved_structures/new_model2.pkl")
	w = model.w 
	iterations =50
	gamma = 0.3
	test_size = 0.01
	repetitions = 4

	results = []
	for rep in range(3,repetitions):
		#train_g,test_g = getFolds(examples_good,test_size,rep)
		train_b,test_b = getFolds(examples_bad,test_size,rep)
		train_g = test_g = examples_good

		model.w = w
		model.reward_f = model.buildRewardFunction()

		learner = Learner(model,train_g,test_g,train_b,test_b)
		#pol,logpol,z = learner.inference(train_g,z_states = None)
		
		learner(iterations,gamma,0.4,examples_type= "good")

		#pickle_saver(learner.results,metrics_directory + "error+lik_" +str(rep)+"_.pkl")
		pickle_saver(learner.model.reward_f,base_directory + "reward_" +str(rep)+"_.pkl",)
		pickle_saver(learner.policy,base_directory + "policy_" +str(rep)+"_.pkl",)
		result = EmptyObject()
		result.final = evaluate_policy(test_g,model,learner.policy)
		result.initial = evaluate_policy(test_g,model,learner.policy_initial)
		results.append(result)
		#name = "Fold %s bad" %idx
		#plot_result(learner.results_b.train_error,learner.results_b.test_error,learner.results_b.train_lik,learner.results_b.test_lik,name)
		#n1 = name+"train" ; n2 = name + "test"
		#trajectoryCompare(train_g,steps,model,n1)	
		#trajectoryCompare(test_b,steps,model,n2)
		#name = "Fold %s good" %idx
		#plot_result(learner.results_g.train_error,learner.results_g.test_error,learner.results_g.train_lik,learner.results_g.test_lik,name)
		#plot_result(learner.results.train_error,learner.results.test_error,learner.results.train_lik,learner.results.test_lik,directory+name)
		#n1 = name+"train" ; n2 = name +"test"
		#trajectoryCompare(train_g,steps,model,learner.policy,learner.policy_initial,plot_directory+n1,plot = True)
		#trajectoryCompare(test_g,steps,model,learner.policy,learner.policy_initial,plot_directory+n2,plot = True)
		#name = "Fold %s bad" %idx
		#n1 = name+"train" ; n2 = name +"test"
		#plot_result(learner.results_b.train_error,learner.results_b.test_error,learner.results_b.train_lik,learner.results_g.test_lik,name)
		#trajectoryCompare(train_b,steps,model,learner.policy,learner.policy_initial,plot_directory+n1,plot=True)
		#trajectoryCompare(test_b,steps,model,learner.policy,learner.policy_initial,plot_directory+n2,plot=True)
	pickle_saver(results,base_directory + "metric.pkl")
		