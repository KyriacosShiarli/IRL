import numpy as np
import math
import string
from discretisationmodel import *
from RFeatures import *
import matplotlib.pyplot as plt
import itertools
from functions import *
import scipy.signal
import scipy.ndimage.filters as filt
from kinematics import inverse_kinematics
import cPickle as pickle


class ExampleStruct(object):
	def __init__(self):
		self.states = []
		self.actions = []
		self.labels = []
		self.state_numbers = []
		self.action_numbers = []
		self.trajectory_number = 0
		self.start = 0
		self.goal = 0
		self.feature_sum = 0
		self.state_action_counts = []
		self.info = ["distance","angle"]

def action_filter(actions,vel_thresh,ang_thresh):
	for n,action in enumerate(actions):
		if n ==0: 
			if np.absolute(action[0]) > ang_thresh and np.absolute(actions[n+1][0]) < ang_thresh:
				action[0] = (0+actions[n+1][0])/2
			elif np.absolute(action[0]) > ang_thresh and np.absolute(actions[n+1][0]) > ang_thresh:
				action[0] = 0
			if np.absolute(action[1]) > vel_thresh and np.absolute(actions[n+1][1]) < vel_thresh:
				action[1] = (0+actions[n+1][1])/2
			elif np.absolute(action[1]) > vel_thresh and np.absolute(actions[n+1][1]) > vel_thresh:
				action[1] = 0
		elif n != len(actions)-1:
			
			if np.absolute(action[0]) > ang_thresh and np.absolute(actions[n+1][0]) < ang_thresh:
				action[0] = (actions[n-1][0]+actions[n+1][0])/2
			elif np.absolute(action[0]) > ang_thresh and np.absolute(actions[n+1][0]) > ang_thresh:
				action[0] = (actions[n-1][0]+actions[n-2][0])/2

			if np.absolute(action[1]) > vel_thresh and np.absolute(actions[n+1][1]) < vel_thresh:
				action[1] = (actions[n-1][1]+actions[n+1][1])/2
			elif np.absolute(action[1]) > vel_thresh and np.absolute(actions[n+1][1]) > vel_thresh:
				action[1] = (actions[n-1][1]+actions[n-2][1])/2
				
		else:
			if np.absolute(action[0]) > ang_thresh:
				action[0] = (actions[n-1][0]+actions[n-2][0])/2

			if np.absolute(action[1]) > vel_thresh:
				action[1] = (actions[n-1][1]+actions[n-2][1])/2
	return actions			




def slice_example(size,example):
	new_examples =[]

	length = example.states.shape[0]
	chunks = length/size
	for i in range(int(chunks)):
		ex = ExampleStruct()

		ex.states = example.states[i*size:i*size+size,:]
		ex.actions = example.actions[i*size:i*size+size,:]
		ex.quality = example.quality
		new_examples.append(ex)
	#print example.states.shape
	#print len(new_examples)
	#print chunks
	#print "--------------------------"
	return new_examples


def load_xy(name):
	def examples_from_stream(f):
		examples = []
		order = []
		for i in range(12):
			print "g"
			line =f.readline()
			if line == '':
				break
			ex = ExampleStruct()
			new = line.split("traj"+str(i+1))
			for i in new[1].replace(' ','').split('||')[1:]:
				al = i.split('|')
				if map(float,al[0].split(','))[1]<50000:
					ex.states.append(map(float,al[0].split(','))[:3])
					ex.actions.append(map(float,al[1].split(',')))
			ex.states = np.array(ex.states); ex.actions = np.array(ex.actions);ex.fake_actions = np.array(ex.actions);
			examples.append(ex)
		return examples
		'''
			ex.actions = action_filter(ex.actions,1,1.5)
			if subsamp == True:
				ex.labels = filt.gaussian_filter(subsample(np.array(ex.labels),8,scal  = 5),1)	
				temp = angle_subsampler(ex.states[:,0],8,scal  = 2)
				temp = np.vstack([temp,subsample(ex.states[:,1],8,scal = 2)])
				temp = np.vstack([temp,angle_subsampler(ex.states[:,2],8,scal = 2)])
				temp = np.vstack([temp,subsample(ex.states[:,3],8,scal = 2)])			
				ex.states = np.transpose(temp)
				#ex.fake_actions = np.zeros([ex.states.shape[0],2]);
				#for k in range(ex.states.shape[0]-1):
				#	ex.fake_actions[k,:] = inverse_kinematics(ex.states[k,:],ex.states[k+1,:],0.129)
				temp1 = subsample(ex.actions[:,0],8,scal = 2)
				temp1 = np.vstack([temp1,subsample(ex.actions[:,1],8,scal = 2)])
				ex.fake_actions = np.transpose(temp1)
				temp = filt.gaussian_filter(scipy.signal.wiener(subsample(ex.actions[:,0],8,scal = 2),25),2)
				temp = np.vstack([temp,filt.gaussian_filter(scipy.signal.wiener(subsample(ex.actions[:,1],8,scal = 2),25),2)])
				ex.actions = np.transpose(temp)
				#ex.actions = ex.fake_actions
			examples.append(ex)	
		'''
	return examples_from_stream(open(name))

if __name__ =="__main__":
	def plot_xy_data(examples):
		base_directory = "data/tests/xy_yaw_filt_extreme/"
		for n,example in enumerate(examples):
			nam = "experiment "+str(n)+"/"
			directory = base_directory + nam
			for i,trajectory in enumerate(example): 
				f,axarr = plt.subplots(3,2,sharex=True)

				make_dir(directory)
				x = trajectory.states.shape[0]
				#print examples[e].actions.shape[0]
				axarr[0,0].scatter(range(x),trajectory.states[:,0],color = "green",alpha = 0.3)
				#axarr[0,0].scatter(range(x),trajectory.states[:,3],color = "blue",alpha = 0.3)
				axarr[0,0].set_ylabel("X")
				#plt.scatter(range(x),e.init,color = "blue",alpha = 0.4)
				#plt.scatter(range(x),e.init,color = "black",alpha = 0.4)
				#plt.savefig(directory+"Distance.png")
				#plt.figure()
				axarr[1,0].scatter(range(x),trajectory.states[:,1],color = "green",alpha = 0.3)
				#axarr[1,0].scatter(range(x),trajectory.states[:,2],color = "blue",alpha = 0.3)
				axarr[1,0].set_ylabel("Y")
				#plt.scatter(range(x),trajectory.init,color = "black",alpha = 0.3)
				#plt.savefig(directory+"Angles.png")
				#plt.figure()
				#axarr[0,1].scatter(range(x),trajectory.fake_actions[:,1],color = "green",alpha = 0.3)
				axarr[0,1].scatter(range(x),trajectory.actions[:,1],color = "blue",alpha = 0.3)
				axarr[0,1].set_ylabel("Angular Velocity")
				#axarr[1,1].scatter(range(x),trajectory.fake_actions[:,0],color = "green",alpha = 0.3)
				axarr[1,1].scatter(range(x),trajectory.actions[:,0],color = "blue",alpha = 0.3)
				axarr[1,1].set_ylabel("Linear Velocity")

				axarr[2,1].scatter(range(x),trajectory.states[:,2],color = "blue",alpha = 0.3)
				axarr[2,1].set_ylabel("YAW")
				
				f.set_size_inches(30.,17.)
				f.savefig(directory+"Original"+str(i)+".png")

	examples_filtered = load_xy("data/Experiments Data Folder v2/PoseAndOrientations_Vels_data/experiment 2/filtered_PoseAndOrientation_Vels_Robot.txt")
	examples_exp2 = load_xy("data/Experiments Data Folder v2/PoseAndOrientations_Vels_data/experiment 2/raw_PoseAndOrientation_Vels_Robot.txt") 
	examples_exp3 = load_xy("data/Experiments Data Folder v2/PoseAndOrientations_Vels_data/experiment 3/raw_PoseAndOrientation_Vels_Robot.txt") 
	ec = load_xy("data/Experiments Data Folder v2/PoseAndOrientations_Vels_data/experiment 2/raw_PoseAndOrientation_Vels_Robot.txt") 
	experiments = [examples_exp2,examples_exp3]
	for experiment in experiments:
		for examp in experiment:
			for i in range(2):
				examp.states[:,i] = filt.gaussian_filter(examp.states[:,i],20)
			examp.states[:,2] = angle_smoother(examp.states[:,2],50)

			for i in range(examp.states.shape[0]-1):
				xvel = (examp.states[i+1,0] - examp.states[i,0])/(0.0167/2)
				yvel = (examp.states[i+1,1] - examp.states[i,1])/(0.0167/2)
				examp.actions[i,1] = np.sqrt(xvel**2 +yvel**2)	
				examp.actions[i,0] = ang_vel(examp.states[i,2],examp.states[i+1,2],0.0167/2)
			examp.actions[-1,:]=examp.actions[-2,:]
	#examples_raw+= examples_filtered

	#with open("data/Experiments Data Folder v2/features_28-11-2014_01.22.03 PM/saved_actions.pkl",'wb')as output:
		#pickle.dump(experiments[0],output,-1)

	#with open("data/Experiments Data Folder v2/features_28-11-2014_02.02.13 PM/saved_actions.pkl",'wb')as output:
	#	pickle.dump(experiments[1],output,-1)
	plot_xy_data([experiments[0]+experiments[1]])
