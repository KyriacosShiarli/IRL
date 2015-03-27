from discretisationmodel import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataload import *
from fg_gradient import adaboost_reg
from kinematics import staticGroupSimple2,staticGroup_with_target
from functions import *

def get_dataset(kinematics,examples):
	next_kin = []
	next_data = []
	for example in examples:
		for n, step in enumerate(example.states[:-1]):
			next_kin.append(kinematics(step,example.actions[n],0.129))
			next_data.append(example.states[n+1])
	next_kin = trajectory_to_cartesian(np.array(next_kin))
	next_data = trajectory_to_cartesian(np.array(next_data))
	diff =0.5*(next_kin-next_data)
	X =np.concatenate([np.hstack((trajectory_to_cartesian(example.states[:-1,:]),example.actions[:-1,:])) for example in examples],axis = 0)	
	return X,diff	

def learn_correction(num_learners,tree_depth):
	def plot_fit():
		f,axarr = plt.subplots(2,sharex=True)
		x = range(diff_test.shape[0])
		axarr[0].plot(x,diff_test[:,0]) 
		axarr[0].plot(x,fit[0,:],color='red',alpha = 0.6)
		axarr[0].set_xlabel("Data Sample")
		axarr[0].set_ylabel("sin(angle difference)")
		axarr[1].plot(x,diff_test[:,1]) 
		axarr[1].plot(x,fit[1,:],color='red',alpha = 0.6)
		axarr[1].set_xlabel("Data Sample")
		axarr[1].set_ylabel("Distance Difference")		

	#learns a transition function from data using adaboost.
	#------------------------------------------------------
	m = DiscModel()
	tot_examples = load_all("Full")
	tot_examples = tot_examples[0] + tot_examples[1]
	train_examples = tot_examples
	test_examples = tot_examples[0:5]
	X_train,diff_train = get_dataset(m.kinematics,train_examples)
	X_test,diff_test = get_dataset(m.kinematics,test_examples)
	
	dimensions = tot_examples[0].states.shape[1]+2
	
	
	estimators = [adaboost_reg(X_train,diff_train[:,i],num_learners,tree_depth) for i in range(dimensions)]
	fit = []
	fit = np.array([estimator.predict(X_test) for estimator in estimators])

	return estimators,examples

def predict_correction(state,action,estimators):
	car_state = state_to_cartesian(state)
	sa = np.hstack((car_state,action))
	return np.concatenate([estimator.predict([sa]) for estimator in estimators])
if __name__ == "__main__":
	#est = learn_tran_regression(20,10)
	#state = np.array([1,2,0.1,0.4])
	#action = np.array([0.3,0.4])
	#out = predict_next(state,action,est)
	#print out
	est = learn_correction(300,10)










