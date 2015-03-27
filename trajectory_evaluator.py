from data_structures import ExampleStruct
import functions as fn
import numpy as np

class Trajectory_Rollout(object):
	#object for the unfoling of policies into trajectories
	def __init__(self,model,policy,state_size=4,action_size=2,supervised = False,continouous=True):
		self.model = model
		self.policy = policy
		self.disc = model.disc
		self.state_size = state_size
		self.action_size = action_size
		self.supervised = supervised
		self.continouous = continouous
	def rollout(self,initial_state,steps):
		trajectory = ExampleStruct()
		trajectory.state_action_counts = np.zeros([self.disc.tot_actions,self.disc.tot_states])
		trajectory.states = np.zeros([steps,self.state_size]);trajectory.actions = np.zeros([steps,self.action_size])
		trajectory.states[0,:] = initial_state
		for i in range(1,steps+1):
			prev_state = self.disc.quantityToState(trajectory.states[i-1,:]);
			action = self.get_next_action(prev_state)
			action_quant =  self.disc.indexToAction(action)
			trajectory.actions[i-1,:] =action_quant
			trajectory.state_action_counts[action,prev_state]+=1
			if i !=steps:
				if self.continouous:
					trajectory.states[i,:]  =self.model.kinematics(trajectory.states[i-1,:],action_quant)
				else:
					keys = self.model.transition.backward[action + self.disc.tot_actions*prev_state].keys()
					values = self.model.transition.backward[action + self.disc.tot_actions*prev_state].values()
					trajectory.states[i,:]  =self.disc.stateToQuantity(np.random.choice(map(int,keys),p=values))
		return trajectory
	def rollout_until(self,initial_state,condition = "x>2.8"):
		trajectory = ExampleStruct()
		trajectory.state_action_counts = np.zeros([self.disc.tot_actions,self.disc.tot_states])
		trajectory.states = np.zeros([1,self.state_size]);trajectory.actions = np.zeros([1,self.action_size])
		trajectory.states[0,:] = initial_state
		steps = 0
		x = trajectory.states[steps,3]
		while eval(condition):
			if steps != 0:
				trajectory.states = np.vstack([trajectory.states,self.model.kinematics(trajectory.states[steps-1,:],action_quant)])
				x = trajectory.states[steps,3]
			prev_state = self.disc.quantityToState(trajectory.states[steps,:]);
			action = self.get_next_action(prev_state)
			action_quant =  self.disc.indexToAction(action)
			trajectory.actions = np.vstack([trajectory.actions,np.zeros([1,self.action_size])])
			trajectory.actions[steps,:] =action_quant
			trajectory.state_action_counts[action,prev_state]+=1
			steps+=1
			if steps > 2000:
				steps == None
				trajectory.steps = -1
				break
		trajectory.steps = steps
		return trajectory
	def get_next_action(self,state):
		if self.supervised == False:
			action = np.random.choice(self.disc.tot_actions,p=self.policy[:,state])
		else:
			row_policy = (self.policy.predict_proba(
			fn.state_to_cartesian(self.disc.stateToQuantity(state))))
			if hasattr(policy,"mapper"):
				classes = mapper	
			else:
				classes = range(len(row_policy[0]))
			action = np.random.choice(classes,p=row_policy[0])
		return action
	def get_intrusions(self,trajectory,bad_states):
		intrusions = 0
		for state in trajectory.states:
			state_number = self.disc.quantityToState(state)
			if np.sum(bad_states==state_number)>0:
				intrusions +=1
		return intrusions
	def direct_evaluation(self,examples,repetitions):
		results = np.zeros([len(examples),2])
		for i,example in enumerate(examples):
			base_traj = np.hstack([example.states,example.actions])
			metrics = np.zeros(repetitions)
			for rep in range(repetitions):			
				agent_trajectory = self.rollout(example.states[0,:],len(example.states))
				ag_traj = np.hstack([agent_trajectory.states,agent_trajectory.actions])
				metrics[rep] = fn.distance_function(base_traj,ag_traj)
			results[i,0] = np.mean(metrics) ; results[i,1] = np.std(metrics)
		return results
	def indirect_evaluation(self,start_states,bad_states,condition,repetitions):
		steps = [] ;intrusions = [];failed = 0 ; results = [0,0,0,0,0]
		for i in range(repetitions):
			start = self.disc.stateToQuantity(np.random.choice(start_states))
			trajectory = self.rollout_until(start,condition)
			if trajectory.steps == None:
				failed+=1
			else:
				steps.append(trajectory.steps)
				intrusions.append(self.get_intrusions(trajectory,bad_states))
		if failed != repetitions:
			results[0] = np.mean(steps);results[1] = np.std(steps)
			results[2] = np.mean(intrusions);results[3] = np.std(intrusions)
		results[4] = failed
		return results



if __name__ == "__main__":
	def test_rollouts():
		print "Loading Model"
		model = fn.pickle_loader("saved_structures/model.pkl")
		##policy = fn.pickle_loader("TESTS/data/test_policy.pkl")
		print "Loading Supervised Policy"
		policy = fn.pickle_loader("saved_structures/supervised_policy.pkl")
		mapper = fn.pickle_loader("saved_structures/supervised_mapper.pkl")
		policy.mapper = mapper
		rollout = Trajectory_Rollout(model,policy,supervised = True)
		init = model.disc.stateToQuantity(455)
		print "Rolling out supervised policy until"
		traj = rollout.rollout_until(init)
		print "Rolling out supervised policy for 30 steps"
		traj = rollout.rollout(init,30)
		print "Loading normal policy"
		policy = fn.pickle_loader("TESTS/data/test_policy.pkl")
		rollout.policy = policy;rollout.supervised = False
		print "Rolling out policy until"
		traj = rollout.rollout_until(init)
		print "Rolling out supervised policy for 30 steps"
		traj = rollout.rollout(init,30)
		print "Done"

	def test_evaluation():
		print "Loading Model"
		model = fn.pickle_loader("saved_structures/model.pkl")
		print "Loading normal policy"
		policy = fn.pickle_loader("TESTS/data/test_policy.pkl")
		rollout = Trajectory_Rollout(model,policy)
		print "Loading examples"
		all_examples = fn.pickle_loader('supervised_cost_function/saved_data.pkl')
		print "Determining bad states"
		bad_states = []
		for example in all_examples:
			for n,label in enumerate(example.labels):
				if label>0.001:
					bad_states.append(model.disc.quantityToState(example.states[n,:]))
		print "Direct Evaluation"
		res = rollout.direct_evaluation(all_examples[:2],1)
		print res
		print "Indirect Evaluation"
		res = rollout.indirect_evaluation([155,455],bad_states,"x>2.8", 5)
		print res
	test_evaluation()



	
