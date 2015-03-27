from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt


def adaboost_class(x,y,estimators,depth):
	bst = AdaBoostClassifier(DecisionTreeClassifier(max_depth = depth),algorithm = "SAMME.R",n_estimators = estimators)
	bst.fit(x,y)
	return bst

def adaboost_reg(x,y,estimators,depth):
	alpha = 1.0
	#reg = svm.SVR()
	#reg = linear_model.Lasso(alpha = 0.1)
	reg = GradientBoostingRegressor(loss='ls',
                                n_estimators=400, max_depth=7,subsample = 0.6,
                              	min_samples_leaf=2,verbose=0)
	#eg = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10),n_estimators = 400)

	reg.fit(x,y)
	return reg

def adaboost_reg2(x,y,estimators,depth):
	alpha = 1.0
	#reg = svm.SVR()
	#reg = linear_model.Lasso(alpha = 0.1)
	reg = GradientBoostingRegressor(loss='ls',
                                n_estimators=estimators, max_depth=depth,subsample = 0.2,verbose=0)
	#eg = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10),n_estimators = 400)
	#print "GOTHERE"
	reg.fit(x,y)
	#print "Go"
	#plt.scatter(range(len(y)),y,color = "red")
	#plt.plot(range(len(y)),reg.predict(x))
	#plt.show()

	return reg

