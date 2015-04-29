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


def slice_example(size,example):
	new_examples =[]
	length = example.states.shape[0]
	chunks = length/size
	for i in range(int(chunks)):
		ex = ExampleStruct()
		ex.states = example.states[-(i*size+size+1):-(i*size+1),:]
		ex.actions = example.actions[-(i*size+size+1):-(i*size+1),:]
		ex.quality = example.quality
		new_examples.append(ex)
		ex.chunk_info = [i+1,chunks]
	#print example.states.shape
	#print len(new_examples)
	#print chunks
	#print "--------------------------"
	return new_examples

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
	#plt.scatter(x,examples_p1[2].states[:,0],color = "green")
	#plt.scatter(x,examples_p2[2].states[:,0],color = "blue")
	#plt.scatter(x,map(angle_half_to_full,examples_p1[2].states[:,0]),color = "yellow")
	#plt.scatter(x,examples_p1[2].com[0,:],color = "black")
	#plt.show()
	return out

def patch_labels(label_vector,patch_size):
	out_vector = np.zeros(len(label_vector))
	label_vector = np.array(label_vector)
	for n,instance in enumerate(label_vector):
		if instance == 1:	
			if n <patch_size:
				out_vector[:n]=1
			elif len(label_vector) - n <patch_size:
				out_vector[n:]=1
			else: 
				out_vector[n-patch_size:n+patch_size] =1
	return out_vector



def loadFile4(name,slicing = "Full",quality_vector=None,subsamp=None):
	def examples_from_stream(f):
		examples = []
		order = []
		#with open(name+'saved_actions.pkl','rb') as input:
		#	action_examples = pickle.load(input)
		for j in range(12):
			line =f.readline()
			if line == '':
				break
			ex = ExampleStruct()
			new = line.split("traj"+str(j+1))
			#ex.trajectory_number = int(line.replace('traj',''))
			for i in new[1].replace(' ','').split('||')[1:]:
			#for i in new[].replace(' ','').split('||')[1:]:
				al = i.split('|')
				if map(float,al[0].split(','))[1]<50000:
					ex.states.append([map(float,al[0].split(','))[0],map(float,al[0].split(','))[1],map(float,al[0].split(','))[5],map(float,al[0].split(','))[4]])
					ex.labels.append(map(float,al[0].split(','))[-1])
					ex.actions.append(map(float,al[1].split(',')))
			ex.states = np.array(ex.states); ex.actions = np.array(ex.actions);ex.fake_actions = np.array(ex.actions);
			#ex.actions = action_filter(ex.actions,1,1.5)
			ex.labels_unpatched = np.zeros(len(ex.labels)) + np.array(ex.labels) 
			ex.labels = patch_labels(ex.labels,15)
			if subsamp != None:
				ex.labels = subsample(np.array(ex.labels),subsamp,scal  = 2)
				temp = angle_subsampler(ex.states[:,0],subsamp,scal  = 0.01)
				temp = np.vstack([temp,subsample(ex.states[:,1],subsamp,scal = 0.01)])
				temp = np.vstack([temp,angle_subsampler(ex.states[:,2],subsamp,scal = 0.01)])
				temp = np.vstack([temp,subsample(ex.states[:,3],subsamp,scal = 0.01)])			
				ex.states = np.transpose(temp)
				#ex.fake_actions = np.zeros([ex.states.shape[0],2]);
				#for k in range(ex.states.shape[0]-1):
				#	ex.fake_actions[k,:] = inverse_kinematics(ex.states[k,:],ex.states[k+1,:],0.129)
				temp1 = subsample(ex.actions[:,0],subsamp,scal = 0.01)
				temp1 = np.vstack([temp1,subsample(ex.actions[:,1],subsamp,scal = 2)])
				ex.fake_actions = np.transpose(temp1)
				#temp = filt.gaussian_filter(scipy.signal.wiener(subsample(ex.actions[:,0],subsamp,scal =   5),101),2)
				temp = np.vstack([temp,subsample(ex.actions[:,1],subsamp,scal = 0.001)])
				ex.actions = np.transpose(temp)
				#ex.actions = ex.fake_actions
			examples.append(ex)
		#def getKey(item):
			#return item.trajectory_number
		#examples = sorted(examples,key = getKey)
		#plt.show()
		return examples		
	# Load all three related files in directory--------------------------------------------------------------------------------
	examples_p1 = examples_from_stream(open(name +"training_samples-person1.txt",'r'))
	examples_p2 = examples_from_stream(open(name +"training_samples-person2.txt",'r'))
	examples_target = examples_from_stream(open(name +"training_samples-person3.txt",'r'))
	#	examples_target = examples_target[::2]
	#plt.scatter(x,examples_p1[2].states[:,0],color = "red")
	#Process to get center of mass for person1 and person2 (the group)---------------------------------------------------------
	for example1, example2,example3 in itertools.izip_longest(examples_p1,examples_p2,examples_target):
		example1.states[:,:2] =  c_o_m([example1.states[:,:2],example2.states[:,:2]])
	#Determine the quality vector---------------------------------------------------------------
	for n,ex in enumerate(examples_p1):
		if quality_vector !=None:
			ex.quality = quality_vector[n]
		else:
			ex.quality = None
	out = []
	#Slice trajectories based on the slicing input------------------------------------
	if slicing !="Full":
		for ex in examples_p1:
			out+=slice_example(slicing,ex)
	else:
		out = examples_p1
	#plt.scatter(x,examples_p1[2].states[:,0],color = "green")
	#plt.scatter(x,examples_p2[2].states[:,0],color = "blue")
	#plt.scatter(x,map(angle_half_to_full,examples_p1[2].states[:,0]),color = "yellow")
	#plt.scatter(x,examples_p1[2].com[0,:],color = "black")
	#plt.show()
	return out


def extract_info(disc_model,num_samples,examples,examples_type = "good"):
	for example in examples:
		example.state_action_counts = np.zeros([disc_model.tot_actions,disc_model.tot_states])
		if num_samples =="Full":
			length = example.states.shape[0]
		else:
			length = num_samples
		example.feature_sum = 0;example.state_numbers = [];example.action_numbers = [];example.steps =length

		for i,state in enumerate(example.states): #Steps to go
			state_number = disc_model.quantityToState(state)
			example.state_numbers.append(state_number)
			example.action_numbers.append(disc_model.actionToIndex(example.actions[i]))
			example.state_action_counts[example.action_numbers[i],state_number]+=1
			if i == 0:
				example.start = state_number
			if i == length-1:
				example.goal = state_number
			example.feature_sum += disc_model.quantityToFeature(state,example.actions[i])
	return examples

def load_all(slicing):
	if slicing =="Full":
		qual = [1,1,0,0,1,0,0,1,1,1,1,0]
		examples1 = loadFile4("data/Experiments Data Folder v2/features_17-11-2014_11.08.31 AM/",slicing = slicing,subsamp = None,quality_vector = qual)
		examples2 = loadFile4("data/Experiments Data Folder v2/features_17-11-2014_04.20.57 PM/",slicing = slicing,subsamp = None,quality_vector = qual)
		#examples1 = [examples1[0], examples1[2], examples1[5], examples1[7]] 
		qual = [1,0,0,1,1,1,0,1,1,0,1,1]
		examples3 = loadFile4("data/Experiments Data Folder v2/features_28-11-2014_01.22.03 PM/",slicing = slicing,subsamp = None,quality_vector = qual) 
		qual = [0,0,0,1,1,1,1,1,1,1,1,1]
		examples4 = loadFile4("data/Experiments Data Folder v2/features_28-11-2014_02.02.13 PM/",slicing = slicing,subsamp = None,quality_vector = qual)
		examples = [examples1,examples2,examples3,examples4]
	else:
		#qual = [1,1,1,1,1,1,1,1,1,1,1,1]
		#examples1 = loadFile4("data/Experiments Data Folder v2/features_17-11-2014_11.08.31 AM/",slicing = slicing,subsamp = 8,quality_vector = qual)
		qual = [1,1,0,0,1,0,0,1,1,1,1,0]
		examples1 = loadFile4("data/Experiments Data Folder v2/features_17-11-2014_11.08.31 AM/",slicing = slicing,subsamp = 8,quality_vector = qual)
		qual = [1,0,0,1,1,1,0,1,1,0,1,1]
		examples3 = loadFile4("data/Experiments Data Folder v2/features_28-11-2014_01.22.03 PM/",slicing = slicing,subsamp = 8,quality_vector = qual) 
		qual = [0,0,0,1,1,1,1,1,1,1,1,1]
		examples4 = loadFile4("data/Experiments Data Folder v2/features_28-11-2014_02.02.13 PM/",slicing = slicing,subsamp = 8,quality_vector = qual)
		examples = examples1+examples3+examples4
	return examples
	
if __name__ =="__main__":
	#examples = loadFile3("data/UPO/Experiments Folder/2014-11-17 11.08.31 AM/")
	#for example in examples:
	#	print "TRAJECTORY NUMBER------------->" example.trajectory_number
	#sa = 0
	#d = DiscModel()
	def plot_labels():
		examples = load_all("Full")
		base_directory = "data/tests/Labels/"
		make_dir(base_directory)
		for n,example in enumerate(examples):
			nam = "experiment "+str(n)
			directory = base_directory + nam
			for i,trajectory in enumerate(example): 
				f = plt.figure()
				ax = f.add_subplot(111)
				ax.scatter(range(len(trajectory.labels)), trajectory.labels,label = "Patched",color = "red", alpha = 0.3)
				ax.scatter(range(len(trajectory.labels)), trajectory.labels_unpatched,color = "blue",label = "Original", alpha = 0.6)
				ax.set_ylabel("Label. 1-0")
				ax.set_xlabel("Time")
				ax.set_xlim([0,len(trajectory.labels)+10])
				ax.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
				f.set_size_inches(10.,7.)
				f.savefig(directory+nam+"Trajectory"+str(i)+".png")

	def plot_raw_data():
		examples = load_all("Full")
		base_directory = "data/tests/all_12.2.2015/"
		for n,example in enumerate(examples):
			nam = "experiment "+str(n)+"/"
			directory = base_directory + nam
			for i,trajectory in enumerate(example): 
				f,axarr = plt.subplots(2,2,sharex=True)

				make_dir(directory)
				x = trajectory.states.shape[0]
				#print examples[e].actions.shape[0]
				axarr[0,0].scatter(range(x),trajectory.states[:,1],color = "green",alpha = 0.3)
				axarr[0,0].scatter(range(x),trajectory.states[:,3],color = "blue",alpha = 0.3)
				axarr[0,0].set_ylabel("Distances")
				#plt.scatter(range(x),e.init,color = "blue",alpha = 0.4)
				#plt.scatter(range(x),e.init,color = "black",alpha = 0.4)
				#plt.savefig(directory+"Distance.png")
				#plt.figure()
				axarr[1,0].scatter(range(x),trajectory.states[:,0],color = "green",alpha = 0.3)
				axarr[1,0].scatter(range(x),trajectory.states[:,2],color = "blue",alpha = 0.3)
				axarr[1,0].set_ylabel("Distances")
				#plt.scatter(range(x),trajectory.init,color = "black",alpha = 0.3)
				#plt.savefig(directory+"Angles.png")
				#plt.figure()
				#axarr[0,1].scatter(range(x),trajectory.fake_actions[:,1],color = "green",alpha = 0.3)
				axarr[0,1].scatter(range(x),trajectory.actions[:,1],color = "blue",alpha = 0.3)
				axarr[0,1].set_ylabel("Linear Velocity")
				axarr[1,1].scatter(range(x),trajectory.fake_actions[:,0],color = "green",alpha = 0.3)
				#axarr[1,1].scatter(range(x),trajectory.actions[:,0],color = "blue",alpha = 0.3)
				axarr[1,1].set_ylabel("Angular Velocity")
				f.set_size_inches(30.,17.)
				f.savefig(directory+"Trajectory"+str(i)+".png")
			#plt.scatter(range(x),e.init2,color = "blue",alpha = 0.4)
	#plot_raw_data()
	m = DiscModel()
	dat = load_all(30)
	dat = extract_info(m,"Full",dat)
	for d in dat:
		print d.feature_sum[16]
		print "QUALITY______",d.quality
 
	#examples_filtered = load_xy("data/Experiments Data Folder v2/PoseAndOrientations_Vels_data/experiment 2/filtered_PoseAndOrientation_Vels_Robot.txt")
	#examples_raw = load_xy("data/Experiments Data Folder v2/PoseAndOrientations_Vels_data/experiment 2/raw_PoseAndOrientation_Vels_Robot.txt")

	
