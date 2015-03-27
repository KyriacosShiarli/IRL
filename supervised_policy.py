from dataload import *
from fg_gradient import adaboost_reg
import sklearn.linear_model as lm
from discretisationmodel import *
from plott import trajectoryCompare_supervised
from discretisationmodel import *
from Model import Model,Transition
import functions as fn
from sklearn import svm
from learn import replace_with_kinematics
from evaluation import evaluate_supervised
# Create predictors that can predict which action to take from the current state only.
# i.e a mapping from state i to action i+1.

# first load the data.

def get_dataset_class(examples):
	X =np.concatenate([np.hstack((example.states[1:,:],example.actions[0:-1,:])) for example in examples],axis = 0)
	y = np.concatenate( [ example.action_numbers[1:] for example in examples]) 
	X = fn.trajectory_to_cartesian(X)
	return X[:,:6],np.array(y)
def get_dataset_cont(examples):
	X =np.concatenate([np.hstack((example.states[1:,:],example.actions[0:-1,:])) for example in examples],axis = 0)
	y1 = np.concatenate( [ example.actions[1:,0] for example in examples])
	y2 = np.concatenate( [ example.actions[1:,1] for example in examples]) 
	X = fn.trajectory_to_cartesian(X)
	return X,np.array(y)	
if __name__ == "__main__":

	disc_model = DiscModel()
	steps =30
	model = fn.pickle_loader("saved_structures/model.pkl")
	examples_good = fn.pickle_loader("saved_structures/examples_good.pkl")
	examples_bad = fn.pickle_loader("saved_structures/examples_bad.pkl")
	train_g,test_g = getFolds(examples_good,0.25,3)

	X_train,y_train = get_dataset_class(train_g)
	mapping = list(set(y_train))
	log_reg = svm.SVC(C = 1.5,kernel = "rbf",gamma = 1.8,probability = True)
	#log_reg = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.3, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
	log_reg.fit(X_train,y_train)
	#T = log_reg.predict_log_proba(X_test)
	results = evaluate_supervised(test_g,model,log_reg,mapping)
	print results
	fn.pickle_saver(results,"results/FeatureFinal/numbers/sup_results.pkl")
	fn.pickle_saver(log_reg,"saved_structures/supervised_policy.pkl")
	fn.pickle_saver(mapping,"saved_structures/supervised_mapper.pkl")

