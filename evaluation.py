import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import distance_function,state_to_cartesian


def trajectory_from_policy(model,policy,initial_state,steps,counts = False):
	disc = model.disc
	if counts == True:
		count = np.zeros([disc.tot_actions,disc.tot_states])
	state_idx = len(disc.bin_info)
	traj = np.zeros([steps,state_idx+2])
	traj[0,:state_idx] = initial_state
	for i in range(1,steps+1):
		prev_state = disc.quantityToState(traj[i-1,:state_idx])
		action = np.random.choice(disc.tot_actions,p=policy[:,prev_state])
		if counts == True: 
			keys = model.transition.backward[action + disc.tot_actions*prev_state].keys()
			values = model.transition.backward[action + disc.tot_actions*prev_state].values()
			count[action,np.random.choice(map(int,keys),p=values)]+=1
		action_quant =  disc.indexToAction(action)
		traj[i-1,state_idx:] =action_quant
		if i !=steps:
			traj[i,:state_idx]  =model.kinematics(traj[i-1,:state_idx],action_quant)
	if counts == False:
		return traj
	else:return count
def counts_from_trajectory(trajectory,discretisation_model,representation = 1):
	if representation == 2:
		traj = np.hstack([trajectory.states,trajectory.actions])
		return counts_from_trajectory(trajectory,discretisation_model,representation=1)
	else:
		out = np.zeros([discretisation_model.tot_actions,discretisation_model.tot_states])
		state_length = len(discretisation_model.bin_info)
		for i in trajectory:
			state = discretisation_model.quantityToState(i[:state_length])
			action = discretisation_model.actionToIndex(i[state_length:])
			out[action,state]+=1
	return out
def evaluate_policy(examples,model,policy,repetitions = 25):
	results = np.zeros([len(examples),2])
	for i,example in enumerate(examples):
		base_traj = np.hstack([example.states,example.actions])
		metrics = np.zeros(repetitions)
		for rep in range(repetitions):			
			agent_trajectory = trajectory_from_policy(model,policy,example.states[0,:],len(example.states))
			metrics[rep] = distance_function(base_traj,agent_trajectory)
		results[i,0] = np.mean(metrics) ; results[i,1] = np.std(metrics)
	return results

def trajectory_from_estimators(model,estimator,mapper,initial_state,steps):
	disc = model.disc
	count = np.zeros([disc.tot_actions,disc.tot_states])
	state_idx = len(disc.bin_info)
	traj = np.zeros([steps,state_idx+2])
	traj[0,:state_idx] = initial_state
	for i in range(1,steps+1):
		policy = estimator.predict_log_proba(state_to_cartesian(traj[i-1,:state_idx]))
		next_action = np.random.choice(mapper,p=np.exp(policy[0]))
		action_quant =  disc.indexToAction(next_action)
		traj[i-1,state_idx:] =action_quant
		if i !=steps:
			traj[i,:state_idx]  =model.kinematics(traj[i-1,:state_idx],traj[i-1,state_idx:])
	return traj

def evaluate_supervised(examples,model,estimator,mapper,repetitions = 25):
	results = np.zeros([len(examples),2])
	for i,example in enumerate(examples):
		base_traj = np.hstack([example.states,example.actions])
		metrics = np.zeros(repetitions)
		for rep in range(repetitions):			
			agent_trajectory = trajectory_from_estimators(model,estimator,mapper,example.states[0,:],len(example.states))
			metrics[rep] = distance_function(base_traj,agent_trajectory)
		results[i,0] = np.mean(metrics) ; results[i,1] = np.std(metrics)
	return results

