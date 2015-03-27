import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from forwardBackward import *
import functions as fn


class EmptyObject(object):
	def __init__(self):
		self.x = None

def plot_result(er_train,er_test,lik_train,lik_test,title):
	numeric,ax = plt.subplots(2,sharex=True)
	ax[0].plot(er_train,c = "r",label = "Train FE Error")
	ax[0].plot(er_test,c = "g",label = "Test FE Error")
	ax[0].set_ylabel("Error")
	ax[1].plot(lik_train,c = "r",label = "Train")
	ax[1].plot(lik_test,c = "g",label = "Test")
	plt.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	ax[1].set_ylabel("Entropy")
	ax[1].set_xlabel("Iterations")
	#numeric.suptitle(title + "Error and Likelihood")
	numeric.savefig(title + " Error and Entropy.png")

	#x = range(progress.shape[1])
	#y = range(progress.shape[0])
	#X,Y = np.meshgrid(x,y)
	#surf = plt.figure()
	#ax2 = surf.add_subplot(111, projection='3d')
	#ax2.plot_surface(X,Y,progress,linewidth = 0,cmap = cm.coolwarm)
	#ax2.set_xlabel("Steps")
	return numeric


def trajectoryCompare(examples,steps,model,policy,policy_initial,name,plot =False,repetitions = 5):
	m = model.disc
	traj = np.zeros([steps,len(m.bin_info)])
	traj_initial = np.zeros([steps,len(m.bin_info)])
	traj_real = np.zeros([steps,len(m.bin_info)])
	x = range(steps)
	if plot == True:
		f,axarr = plt.subplots(2,2,sharex=False)
	for num,example in enumerate(examples[:10]):
		for rep in range(repetitions):
			length = example.steps
			traj[0,:] = example.states[-length]
			traj_initial[0,:] = example.states[-length]
			traj_real[0,:] = example.states[-length]
			for i in range(1,length):
				prev_state = m.quantityToState(traj[i-1,:])
				prev_state_initial = m.quantityToState(traj_initial[i-1,:])
				action = np.random.choice(m.tot_actions,p=policy[:,prev_state])

				#action = np.argmax(policy[:,prev_state])
				action_initial = np.random.choice(m.tot_actions,p=policy_initial[:,prev_state_initial])
				#action_initial = np.argmax(policy[:,prev_state_initial])
				action_quant = model.disc.indexToAction(action)
				action_quant_initial = model.disc.indexToAction(action_initial)
				#keys = map(int,model.transition.backward[action + prev_state*m.tot_actions].keys())
				#val = model.transition.backward[action + prev_state*m.tot_actions].values()
				traj[i,:]  =model.kinematics(traj[i-1,:],action_quant)
				traj_initial[i,:]  =model.kinematics(traj_initial[i-1,:],action_quant_initial)
				traj_real[i,:]   = model.kinematics(traj_real[i-1,:],example.actions[-length+i])
			#distance_metric_initial[num,rep] = distance_funtion(traj_real,traj_initial)
			#distance_metric_final[num,rep] = distance_funtion(traj_real,traj)
			if plot == True:
				if len(examples)>1:
					intensity = 0.3
				else:
					intensity = 0.7
				axarr[0,0].scatter(x,traj_real[:,1],color = "red",alpha = intensity)
				axarr[0,1].scatter(x,traj_real[:,0],color = "red",alpha = intensity)
				axarr[1,0].scatter(x,traj_real[:,2],color = "red",alpha = intensity)
				axarr[1,1].scatter(x,traj_real[:,3],color = "red",alpha = intensity)
				axarr[0,0].scatter(x,traj[:,1],color = "green",label = "learned",alpha = intensity)
				axarr[0,1].scatter(x,traj[:,0],color = "green",alpha = intensity)
				axarr[1,0].scatter(x,traj[:,2],color = "green",label = "learned",alpha = intensity)
				axarr[1,1].scatter(x,traj[:,3],color = "green",alpha = intensity)

				axarr[0,0].scatter(x,traj_initial[:,1],color = "blue",label = "learned",alpha = intensity)
				axarr[0,1].scatter(x,traj_initial[:,0],color = "blue",alpha = intensity/3)
				axarr[1,0].scatter(x,traj_initial[:,2],color = "blue",label = "learned",alpha = intensity)
				axarr[1,1].scatter(x,traj_initial[:,3],color = "blue",alpha = intensity/3)		
	if plot == True:		
		axarr[0,0].set_ylabel("Distance")
		axarr[0,1].set_ylabel("Angle")
		axarr[1,0].set_ylabel("Target Angle")
		axarr[1,1].set_ylabel("Target Distance")
		f.set_size_inches(22.,17.)
		f.savefig(name+str(num)+".png",dpi=80)
def trajectoryCompare_supervised(examples,estimator,steps,model,name):
	m = model.disc
	traj = np.zeros([steps,4])
	x = range(steps)
	repetitions = 5
	f,axarr = plt.subplots(4,sharex=True)
	for num,example in enumerate(examples):
		for rep in range(repetitions):
			length = steps
			traj[0,:] = example.states[-length]
			prev_action = example.actions[-length]
			for i in range(1,length):
				ex = np.hstack([traj[i-1,:],prev_action])
				print "EX", ex
				policy = np.exp(estimator.predict_log_proba(ex)[0])
				print sum(policy),policy
				action = np.random.choice(m.tot_actions,p=policy)
				action_quant = m.indexToAction(action)
				traj[i,:]  =model.kinematics(traj[i-1,:],action_quant)
				prev_action = action_quant
				if rep==0:
					axarr[0].scatter(i,example.states[-length+i][1],color = "red",alpha = 0.5)
					axarr[1].scatter(i,example.states[-length+i][0],color = "red",alpha = 0.5)
					axarr[2].scatter(i,example.states[-length+i][2],color = "red",alpha = 0.5)
					axarr[3].scatter(i,example.states[-length+i][3],color = "red",alpha = 0.5)
			axarr[0].scatter(x,traj[:,1],color = "green",label = "learned",alpha = 0.3)
			axarr[1].scatter(x,traj[:,0],color = "green",alpha = 0.3)
			axarr[2].scatter(x,traj[:,2],color = "green",label = "learned",alpha = 0.3)
			axarr[3].scatter(x,traj[:,3],color = "green",alpha = 0.3)
		axarr[0].set_ylabel("Distance")
		axarr[1].set_ylabel("Angle")
		axarr[3].set_ylabel("Target Distance")
		axarr[2].set_ylabel("Target Angle")
		f.savefig("results3/"+name+str(num)+".png")
'''
def trajectoryCompare(examples,steps,model,name):
	m = model.disc
	traj = np.zeros([steps,len(m.bin_info)])
	f,axarr = plt.subplots(4,sharex=True)
	x = range(steps)
	goals = [(ex.goal) for ex in examples]
	policy,log_policy,z_states= caus_ent_backward(model.transition,model.reward_f,goals,steps) 
	for example in examples:
		traj[0,:] = example.states[0]
		for i in range(1,example.steps):
			prev_state = m.quantityToState(traj[i-1,:])
			action = np.random.choice(m.tot_actions,p=policy[:,prev_state])
			#keys = map(int,model.transition.backward[action + prev_state*m.tot_actions].keys())
			#val = model.transition.backward[action + prev_state*m.tot_actions].values()
			next_state =np.random.choice(keys,p=val) 
			#plt.plot(policy[:,prev_state])
			#plt.scatter(action,policy[action,prev_state])
			#plt.show()
			traj[i,:] = m.stateToQuantity(next_state)
			axarr[0].scatter(i,m.stateToQuantity(example.state_numbers[i])[1],color = "red",alpha = 0.3)
			axarr[1].scatter(i,m.stateToQuantity(example.state_numbers[i])[0],color = "red",alpha = 0.3)
		axarr[0].scatter(x,traj[:,1],color = "green",label = "learned",alpha = 0.3)
		axarr[1].scatter(x,traj[:,0],color = "green",alpha = 0.3)		
	axarr[0].set_ylabel("Distance")
	axarr[1].set_ylabel("Angle")
	f.savefig(name+".png")
'''


def plot_metric_results():
	base_directory = "results/"
	result_sets = ["NormalFinal/","FeatureFinal/"]
	rest = "numbers/metric.pkl"
	metric = fn.pickle_loader(base_directory+result_sets[0]+rest)
	print metric[0].initial - metric[0].final

	metric2 = fn.pickle_loader(base_directory+result_sets[1]+rest)
	metric_sup = fn.pickle_loader(base_directory+result_sets[1]+"numbers/sup_results.pkl")
	print metric[0].initial - metric[0].final
	metric_loss_aug = fn.pickle_loader(base_directory+"LossaugmentedFinal/"+"numbers/metric.pkl")
	metric_naive = fn.pickle_loader(base_directory+"LossNaiveFinal/"+"numbers/metric.pkl")
	print metric_loss_aug[0].final
	all_initials = metric[1].initial
	all_finals_normal = metric[1].final
	all_finals_feature = metric2[1].final
	loss_aug = metric_loss_aug[0].final
	loss_naive = metric_naive[0].final
	x = np.array(range(len(all_initials[0:25])))
	f = plt.figure()
	ax = f.add_subplot(111)
	width = 0.15
	ax.bar(x-2*width,all_initials[0:25,0],width,yerr = all_initials[0:25,1],color = "r",ecolor = "r",label = "Random")
	ax.bar(x-width,all_finals_normal[:25,0],width,yerr = all_finals_normal[:25,1],color = "b",ecolor = "b",label = "Full")
	ax.bar(x,metric_sup[:25,0],width,yerr = metric_sup[:25,1],color = "black",ecolor = "black",label = "Imitation Learning")
	ax.bar(x+1*width,all_finals_feature[:25,0],width,yerr = all_finals_feature[:25,1],color = "green",ecolor = "green",label = "Features")
	#ax.bar(x+0*width,loss_naive[:25,0],width,yerr = loss_aug[:25,1],color = "b",ecolor = "blue",label = "Naive")
	#ax.bar(x+1*width,loss_aug[:25,0],width,yerr = loss_aug[:25,1],color = "red",ecolor = "red",label = "Loss Augmented")

	ax.set_xlim(-0.30,25.15)
	ax.xaxis.set_ticks(range(0,len(x)))
	ax.grid(b=True,which="both",axis="x")
	ax.set_ylabel("Distance from Original",fontweight = 'bold',fontsize = 17)
	ax.set_xlabel("CV Examples",fontweight='bold',fontsize = 17)
	ax.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	f.set_size_inches(25.,17.)
	f.savefig(base_directory+result_sets[0]+"plots/"+"metricCompare.png")


	x = np.array([2,5,8,11,14,17])
	f = plt.figure()
	ax = f.add_subplot(111)
	width = 0.60
	ax.set_xticklabels(("Random","Full","Supervised","Feature","Naive","Loss Augmented"),fontsize = 15)
	(ax.bar(x-width/2,[np.sum(all_initials[0:25,0])/25,np.sum(all_finals_normal[:25,0])/25,np.sum(metric_sup[:25,0])/25,np.sum(all_finals_feature[:25,0])/25,np.sum(loss_naive[0:25,0])/25,np.sum(loss_aug[0:25,0])/25],
		width, yerr = [np.sum(all_initials[0:25,1])/25,np.sum(all_finals_normal[:25,1])/25,np.sum(metric_sup[:25,1])/25,np.sum(all_finals_feature[:25,1])/25,np.sum(loss_naive[0:25,1])/25,np.sum(loss_aug[0:25,1])/25],
		color = ["r","b","black","g","m","y"],ecolor = "r"))
	#ax.bar(x,np.sum(all_finals_normal[:25,0])/25,width,yerr = np.sum(all_finals_normal[:25,1])/25,color = "b",ecolor = "b",label = "Full")
	#ax.bar(x,np.sum(metric_sup[:25,0])/25,width,yerr = np.sum(metric_sup[:25,1])/25,color = "black",ecolor = "black",label = "Supervised")
	#ax.bar(x+0*width,np.sum(all_finals_feature[:25,0])/25,width,yerr = np.sum(all_finals_feature[:25,1])/25,color = "green",ecolor = "green",label = "Features")
	#ax.bar(x+1*width,loss_aug[:25,0],width,yerr = loss_aug[:25,1],color = "y",ecolor = "green",label = "Loss Augmented")
	ax.xaxis.set_ticks(x)
	ax.grid(b=True,which="both",axis="x")
	ax.set_ylabel("Average Euclidian Distance",fontweight = 'bold',fontsize = 17)
	ax.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	f.set_size_inches(15.,6.)
	f.savefig(base_directory+result_sets[0]+"plots/"+"metricCompareAvg.png")


	x = np.array([2,4,6,8])
	f = plt.figure()
	ax = f.add_subplot(111)
	width = 0.60
	ax.set_xticklabels(("Imitation Learning","Feature","Naive","Loss Augmented"),fontweight = 'bold',fontsize = 20)
	(ax.bar(x-width/2,[np.sum(metric_sup[:25,0])/25,np.sum(all_finals_feature[:25,0])/25,np.sum(loss_naive[0:25,0])/25,np.sum(loss_aug[0:25,0])/25],
		width, yerr = [np.sum(metric_sup[:25,1])/25,np.sum(all_finals_feature[:25,1])/25,np.sum(loss_naive[0:25,1])/25,np.sum(loss_aug[0:25,1])/25],color = ["black","g","b","r"],ecolor = "r"))
	#ax.bar(x,np.sum(all_finals_normal[:25,0])/25,width,yerr = np.sum(all_finals_normal[:25,1])/25,color = "b",ecolor = "b",label = "Full")
	#ax.bar(x,np.sum(metric_sup[:25,0])/25,width,yerr = np.sum(metric_sup[:25,1])/25,color = "black",ecolor = "black",label = "Supervised")
	#ax.bar(x+0*width,np.sum(all_finals_feature[:25,0])/25,width,yerr = np.sum(all_finals_feature[:25,1])/25,color = "green",ecolor = "green",label = "Features")
	#ax.bar(x+1*width,loss_aug[:25,0],width,yerr = loss_aug[:25,1],color = "y",ecolor = "green",label = "Loss Augmented")
	ax.xaxis.set_ticks(x)
	ax.grid(b=True,which="both",axis="x")
	ax.set_ylabel("Average Euclidian Distance",fontweight = 'bold',fontsize = 20)
	ax.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	f.set_size_inches(15.,9.)
	f.savefig(base_directory+result_sets[0]+"plots/"+"metricCompareAvgLA.png")
	



if __name__ == "__main__":
	plot_metric_results()

