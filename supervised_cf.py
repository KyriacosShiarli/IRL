from dataload import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import *
from discretisation import *
from sklearn import svm
import sklearn.cross_validation as crosval
import cPickle as pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from discretisationmodel import DiscModel
#Build a cost function based on the labels in the data.

######################## REQUIREMENTS
# 1. Load data with the labels.
# 2. Discretise the labels to create 3 or 4 classes.
# 3. Visualise the data. and classes
# 4. Biasing problem
# 5. What classifiers?
# 6. Cross validation methods?
# 7. Plots?
#########################################################



	
class results_structure():
	def __init__(self,estimator,score,score_avg,weight,gamma,C):
		self.estimator = estimator
		self.score = score
		self.score_avg = score_avg		
		
def two_class_metric(y_original,y_predicted):
	neg = y_original - y_predicted
	pcnt_false_neg = float(len(np.where(neg == 1)[0]))/len(np.where(y_original == 1)[0])
	pcnt_false_pos = float(len(np.where(neg == -1)[0]))/len(np.where(y_original == 0)[0])
	print pcnt_false_neg,pcnt_false_pos
	return pcnt_false_pos,pcnt_false_neg

def custom_score(false_possitives,false_negatives,alpha):
	return (alpha*(1-false_possitives) + (1-false_negatives))/(1+alpha)

def custom_score2(false_possitives,false_negatives,alpha):
	return ((np.exp(1-false_possitives)**alpha)*(1-false_negatives))/(np.exp(1)**alpha)

def load_and_concatenate(save):
	if save ==True:
		experiments = load_all("Full")
		all_data = experiments[0]+experiments[2]+experiments[3]
		with open('supervised_cost_function/saved_data.pkl','wb')as output:
			pickle.dump(all_data,output,-1)
	else:
		with open('supervised_cost_function/saved_data.pkl','rb') as input:
			all_data = pickle.load(input)
	return all_data

def plot_label_patching(all_examples):
	base_directory = "supervised_cost_function/Preprocessing/LabelSubsampling/"
	make_dir(base_directory)
	# make a histogram of the label values so you can discretise
	for i,example in enumerate(all_examples):
		plt.figure()
		x = range(len(example.labels))
		plt.scatter(x,example.labels,alpha = 0.3)
		plt.savefig(base_directory+"Experiment"+str(i)+"Example"+"test"+".png")
	plt.show()

def split_in_classes(class_intervals,all_examples):
	counts = np.zeros(len(class_intervals))
	for n,example in enumerate(all_examples):
		for k,label in enumerate(example.labels):
			if label>0.001:
				example.labels[k] = 1
				counts[example.labels[k]]+=1
	return all_examples

def visualise_classes(all_examples):
	directory = "supervised_cost_function/Preprocessing/LabelPlots/"
	make_dir(directory)
	f,axarr = plt.subplots(3,sharex=False)

	for example in all_examples:
		for i,label in enumerate(example.labels):
			if label ==0:
				axarr[0].scatter(example.states[i,0],example.states[i,1],color = "blue",alpha = 0.2)
				axarr[1].scatter(example.states[i,2],example.states[i,3],color = "blue",alpha = 0.2)
				axarr[2].scatter(example.actions[i,0],example.actions[i,1],color = "blue",alpha = 0.2)
			elif label ==1:
				axarr[0].scatter(example.states[i,0],example.states[i,1],color = "green",alpha = 0.8)
				axarr[1].scatter(example.states[i,2],example.states[i,3],color = "green",alpha = 0.8)
				axarr[2].scatter(example.actions[i,0],example.actions[i,1],color = "green",alpha = 0.8)
			elif label ==2:
				axarr[0].scatter(example.states[i,0],example.states[i,1],color = "red",alpha = 0.8)
				axarr[1].scatter(example.states[i,2],example.states[i,3],color = "red",alpha = 0.8)
				axarr[2].scatter(example.actions[i,0],example.actions[i,1],color = "red",alpha = 0.8)
			elif label ==3:
				axarr[0].scatter(example.states[i,0],example.states[i,1],color = "black",alpha = 0.8)
				axarr[1].scatter(example.states[i,2],example.states[i,3],color = "black",alpha = 0.8)
				axarr[2].scatter(example.actions[i,0],example.actions[i,1],color = "black",alpha = 0.8)
	f.set_size_inches(30.,17.)
	axarr[0].set_xlabel ="Group Angle";axarr[1].set_ylabel ="Group Distance"
	axarr[1].set_xlabel ="Target Angle";axarr[1].set_ylabel ="Target Distance"
	axarr[2].set_xlabel ="Robot Angular velocity";axarr[2].set_ylabel ="Robot linear Velocity" 
	f.savefig(directory+"LabelPlots3Pairs2Class.png")

def visualise_xy(X,y):
	directory = "supervised_cost_function/Preprocessing/LabelPlots/"
	make_dir(directory)
	f,axarr = plt.subplots(3,sharex=False)
	for n,predicted_label in enumerate(y):
		if predicted_label ==0:
			axarr[0].scatter(X[n,0],X[n,1],color = "blue",alpha = 0.2)
			axarr[1].scatter(X[n,2],X[n,3],color = "blue",alpha = 0.2)
			axarr[2].scatter(X[n,4],X[n,5],color = "blue",alpha = 0.2)
		elif predicted_label ==1:
			axarr[0].scatter(X[n,0],X[n,1],color = "black",alpha = 0.8)
			axarr[1].scatter(X[n,2],X[n,3],color = "black",alpha = 0.8)
			axarr[2].scatter(X[n,4],X[n,5],color = "black",alpha = 0.8)
		elif predicted_label ==2:
			axarr[0].scatter(example.states[i,0],example.states[i,1],color = "red",alpha = 0.8)
			axarr[1].scatter(example.states[i,2],example.states[i,3],color = "red",alpha = 0.8)
			axarr[2].scatter(example.actions[i,0],example.actions[i,1],color = "red",alpha = 0.8)
		elif predicted_label ==3:
			axarr[0].scatter(example.states[i,0],example.states[i,1],color = "black",alpha = 0.8)
			axarr[1].scatter(example.states[i,2],example.states[i,3],color = "black",alpha = 0.8)
			axarr[2].scatter(example.actions[i,0],example.actions[i,1],color = "black",alpha = 0.8)
	f.set_size_inches(30.,17.)
	axarr[0].set_xlabel ="Group Angle";axarr[1].set_ylabel ="Group Distance"
	axarr[1].set_xlabel ="Target Angle";axarr[1].set_ylabel ="Target Distance"
	axarr[2].set_xlabel ="Robot Angular velocity";axarr[2].set_ylabel ="Robot linear Velocity" 
	f.savefig(directory+"LabelPlotsPredicted.png")


def visualise_metric():
	directory = "supervised_cost_function/"
	make_dir(directory)
	f = plt.figure()
	ax = f.add_subplot(111,projection = "3d")
	fp  = np.linspace(0,1,50)
	fn  = np.linspace(0,1,50)
	X,Y = np.meshgrid(fp,fn)
	z = np.array([custom_score(x,y,2)for x,y in zip(np.ravel(X),np.ravel(Y))])
	z_alt = np.array([custom_score2(x,y,2)for x,y in zip(np.ravel(X),np.ravel(Y))])
	Z = z.reshape(X.shape)
	Z_alt = z_alt.reshape(X.shape)
	#ax.plot_surface(X,Y,Z,label = "Ml")
	ax.plot_surface(X,Y,Z_alt,cmap ='BuGn',label = "Me" )
	ax.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	ax.set_ylabel("False Negatives")
	ax.set_xlabel("False Positives")
	ax.set_zlabel("Score")
	f.savefig(directory+"supervised_metric.png")
	plt.show()

def get_dataset(all_examples):
	for example in all_examples:
		x1,y1 = polar_to_cartesian(example.states[:,0],np.ones(example.states.shape[0]))
		pol1 = np.vstack([x1,y1])
		print "LEN", len(x1)
		xt,yt = polar_to_cartesian(example.states[:,2],np.ones(example.states.shape[0]))
		polt = np.vstack([pol1,example.states[:,1],xt,yt,example.states[:,3]])
		print "LEN", len(x1)
		print example.states.shape[0]
		ex = np.hstack([np.transpose(polt),example.actions])
		if "X" in locals():
			X = np.concatenate((X,ex),axis = 0)
			y = np.concatenate((y,example.labels),axis = 0)
		else:
			X = ex
			y = example.labels
	return X,y

def get_dataset_only_state(all_examples):
	for example in all_examples:
		x1,y1 = polar_to_cartesian(example.states[:,0],np.ones(example.states.shape[0]))
		pol1 = np.vstack([x1,y1])
		print "LEN", len(x1)
		xt,yt = polar_to_cartesian(example.states[:,2],np.ones(example.states.shape[0]))
		polt = np.vstack([pol1,example.states[:,1],xt,yt,example.states[:,3]])
		print "LEN", len(x1)
		#print example.states.shape[0]
		ex = np.transpose(polt)
		#print ex
		if "X" in locals():
			X = np.concatenate((X,ex),axis = 0)
			y = np.concatenate((y,example.labels),axis = 0)
		else:
			X = ex
			y = example.labels
	return X,y

def get_dataset_discrete_state(all_examples,disc_model):
	for example in all_examples:
		for state in example.states:
			st = disc_model.quantityToState(state)
			state = disc_model.stateToQuantity(st)
			cart  = state_to_cartesian(state)
			if "X" in locals():
				X = np.vstack([X,cart])
			else:
				X = cart
		if "y" in locals():
			y = np.concatenate((y,example.labels),axis = 0)
		else:
			y = example.labels
	return X,y


def shuffle_cross_validation(estimator,X,y):
	number = 6
	score = []
	ss = crosval.ShuffleSplit(len(y),n_iter=number,test_size = 0.25,random_state=0)
	for train_index,test_index in ss:
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]
		estimator.fit(X_train,y_train)
		y_predict = estimator.predict(X_test)
		fp,fn = two_class_metric(y_test,y_predict)
		score.append(custom_score2(fp,fn,3))
	return np.array(score),np.sum(score)/number,np.std(np.array(score))

def qualitative_plot(X,y):
	directory = "supervised_cost_function/AlgorithmCompare/LabelPlots/"
	make_dir(directory)
	ss = crosval.ShuffleSplit(len(y),n_iter=1,test_size = 0.25,random_state=0)
	with open('eses.pkl','rb') as input:
		ss = pickle.load(input)
	estimator = svm.SVC(C = 2.0,kernel = "rbf",class_weight = {0:1.1,1:1},gamma = 6)
	for train_index,test_index in ss:
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]
		estimator.fit(X_train,y_train)
		y_predict = estimator.predict(X_test)
		f,axarr = plt.subplots(2,sharex=False)
		i=0
		for label_predict,label_test in zip(y_predict,y_test):
			x_test = state_to_polar(X_test[i,:6])
			if label_test ==0 and label_predict ==0:
				axarr[0].scatter(x_test[0],x_test[1],color = "blue",alpha = 0.3)
				axarr[1].scatter(x_test[2],x_test[3],color = "blue",alpha = 0.3)
				#axarr[2].scatter(X_test[i,6],X_test[i,7],color = "blue",alpha = 0.3)
			if label_predict ==1 and label_test ==1:
				axarr[0].scatter(x_test[0],x_test[1],color = "black",alpha = 0.6)
				axarr[1].scatter(x_test[2],x_test[3],color = "black",alpha = 0.6)
				#axarr[2].scatter(X_test[i,6],X_test[i,7],color = "black",alpha = 0.6)
			if label_test ==1 and  label_predict==0:
				axarr[0].scatter(x_test[0],x_test[1],color = "red",alpha = 0.6)
				axarr[1].scatter(x_test[2],x_test[3],color = "red",alpha = 0.6)
				#axarr[2].scatter(X_test[i,6],X_test[i,7],color = "red",alpha = 0.6)
			if label_test ==0 and  label_predict==1:
				axarr[0].scatter(x_test[0],x_test[1],color = "green",alpha = 0.6)
				axarr[1].scatter(x_test[2],x_test[3],color = "green",alpha = 0.6)
				#axarr[2].scatter(X_test[i,6],X_test[i,7],color = "green",alpha = 0.6)
			i+=1
		axarr[0].scatter(0,0,label = "Correct -" ,color = "blue",alpha = 0.7)
		axarr[0].scatter(0,0,label = "Correct +" ,color = "black",alpha = 0.7)
		axarr[0].scatter(0,0,label = "False  +" ,color = "green",alpha = 0.7)
		axarr[0].scatter(0,0,label = "False -" ,color = "red",alpha = 0.7)
		axarr[0].scatter(0,0,color = "white",alpha = 1)
		axarr[0].legend(bbox_to_anchor=(0.55, 1,0.5,-0.06),loc=1)	
		axarr[0].set_xlabel("Angle From Group",fontweight = 'bold',fontsize = 15)
		axarr[0].set_ylabel("Distance From Group",fontweight = 'bold',fontsize = 15)
		axarr[1].set_xlabel("Angle From Target",fontweight = 'bold',fontsize = 15)
		axarr[1].set_ylabel("Distance From Target",fontweight = 'bold',fontsize = 15)	
		f.set_size_inches(15.,12.)
		f.savefig(directory+"high.png")
def grid_search_svms(X,y):
	c_vector = [1.8,2.0]
	gamma_vector = np.arange(7.0,10.0,1.)
	weight_vector = [0.04,0.06,0.07]
	all_results =[]
	num_results = []
	std = []
	for k in c_vector:
		for j in gamma_vector:
			for i in weight_vector:
				clf = svm.SVC(C = k,kernel = "rbf",class_weight = {0:i,1:1},gamma = j)
				print "WEIGHT----------------------",i
				print "Gamma----------------------",j
				print "C----------------------",k
				score,avg_score,st = shuffle_cross_validation(clf,X,y)
				all_results.append(results_structure(clf,score,avg_score,i,j,k))
				num_results.append(avg_score)
				std.append(st)
				print score,avg_score
	return all_results,num_results,std

def grid_search_adaboost(X,y):		
	depth_vector = np.arange(7,11,1)
	number_of_estimators = np.arange(300,500,50) 
	all_results =[]
	num_results = []
	std = []
	for k in depth_vector:
		for j in number_of_estimators:
			clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = k),algorithm = "SAMME.R",n_estimators = j)
			print "Depths----------------------",k
			print "Estimators----------------------",j
			score,avg_score,st = shuffle_cross_validation(clf,X,y)
			all_results.append(results_structure(clf,score,avg_score,i,j,k))
			std.append(st)
			num_results.append(avg_score)
			print score,avg_score
	return all_results,num_results,std

def grid_search_gboost(X,y):		
	depth_vector = np.arange(7,11,1)
	number_of_estimators = np.arange(300,500,50) 
	all_results =[]
	num_results = []
	std = []
	for k in depth_vector:
		for j in number_of_estimators:
			clf =GradientBoostingClassifier(
                              	n_estimators=j, max_depth=k,subsample = 0.8,
                               	min_samples_leaf=2,
                                min_samples_split=5,verbose=0)
			print "Depths----------------------",k
			print "Estimators----------------------",j
			score,avg_score,st = shuffle_cross_validation(clf,X,y)
			all_results.append(results_structure(clf,score,avg_score,i,j,k))
			std.append(st)
			num_results.append(avg_score)
			print score,avg_score
	return all_results,num_results,std

def algorithm_compare():
	directory = "supervised_cost_function/AlgorithmCompare/"
	make_dir(directory)			
	all_examples = load_and_concatenate(False)
	#plot_label_patching(all_examples)
	all_examples = split_in_classes(np.array([0,0.001]),all_examples)
	visualise_classes(all_examples)
	m = DiscModel()
	X,y = get_dataset_only_state(all_examples)

	all_gboost,num_gboost,std_boost = grid_search_gboost(X,y)
	all_adaboost,num_adaboost,std_ada = grid_search_adaboost(X,y)
	all_svm,num_svm,std_svm = grid_search_svms(X,y)
	with open('supervised_cost_function/saved_alcompare.pkl','wb')as output:
		pickle.dump([all_gboost,num_gboost,all_adaboost,num_adaboost,all_svm,num_svm],output,-1)


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.errorbar(range(len(num_gboost)),np.array(num_gboost),yerr =std_boost,fmt='o',color = "red",label = "Gradient Boosting")
	ax.errorbar(range(len(num_adaboost)),np.array(num_adaboost),yerr =std_ada,fmt='o',color = "blue",label = "AdaBoost")
	ax.errorbar(range(len(num_svm)),np.array(num_svm),yerr =std_svm,fmt='o',color = "green", label = "Biased Svm")
	ax.set_xlim(-1,len(num_svm)+15)
	fig.set_size_inches(15.,12.)
	ax.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	ax.set_ylabel("Me")
	ax.set_xlabel("Hyperparameter Variations")
	fig.savefig(directory + "SVMADABOOSTGBOOST_CrossValidated_Gridsearched.png")

#visualise_metric()
#algorithm_compare()

disc = DiscModel()
all_examples = load_and_concatenate(False)
all_examples = split_in_classes(np.array([0,0.001]),all_examples)
X,y = get_dataset_discrete_state(all_examples,disc)
#Xalt,yalt = get_dataset_alt(all_examples)

#qualitative_plot(X,y)
#algorithm_compare()

#with open('supervised_cost_function/saved_alcompare.pkl','rb')as input:
#	results = pickle.load(input)
#print results

#visualise_metric()
estimator = svm.SVC(C = 2.,kernel = "rbf",class_weight = {0:0.06,1:1},gamma = 8.0,probability = True)
estimator.fit(X,y)
print np.amax(y)
te =estimator.predict(X)
p,n = two_class_metric(y,te)
out = custom_score2(p,n,2)
print out
m = DiscModel()
supervised_reward = np.zeros([m.tot_actions,m.tot_states])
for j in range(m.tot_states):
	st = m.stateToQuantity(j)
	st = state_to_cartesian(st)
	#print estimator.predict([st])
	
	supervised_reward[:,j] = estimator.predict([st])[0]
	#print supervised_reward[i,j]
pickle_saver(supervised_reward,"saved_structures/supervised_reward_state2.pkl")

#cv = crosval.cross_val_score(clf,X,y,cv = 3,scoring = "mean_squared_error")
#clf.fit(X,y)
#Y = clf.predict(X)
#print "Maximum",np.amax(X)
#print sum(Y) 
#visualise_xy(X,y)

#fp,fn = two_class_metric(y,Y)
#print fp,fn
#out = custom_score(fp,fn)
#print out


#clf.fit(X,y)
#visualise_metric()





