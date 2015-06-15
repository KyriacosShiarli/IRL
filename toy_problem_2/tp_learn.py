import tp_forwardBackward as fb
from tp_Model import *
from tp_discretisationmodel import *
import tp_functions as fn
from tp_evaluation import *
import scipy.stats as sps
from tp_data_structures import EmptyObject

def pin_to_zero(vector_to_pin):
	vector_to_pin = np.array(vector_to_pin)
	where = [i for i in range(len(vector_to_pin)) if vector_to_pin[i] < 0]
	vector_to_pin[where]= 0
	return vector_to_pin

def alpha_check(rho,theta_1,theta_2,D,theta_3):
	alpha_plus = (apprentice.zeta-D+theta_1)/theta_3
	alpha_minus = (-apprentice.zeta-D+theta_2)/theta_3
	for n,i in alpha_plus:
		if i<0:
			v=5
def inference(model,steps,initial_states,discount = 0.9,z_states=None):
	policy,log_policy,z_states=fb.caus_ent_backward(model.transition,model.reward_f,discount = discount,conv=5,z_states = z_states)
	state_freq,state_action_freq,all_stat = fb.forward(policy,model.transition,initial_states,steps,discount=discount) # 30 timesteps
	return policy,z_states,state_action_freq,all_stat
def plot_path(path_probs,disc_model,subplot_dims,save_name):
	fig,axarr = plt.subplots(subplot_dims[0],subplot_dims[1],sharex = False)
	all_states = disc_model.enum_states
	steps = len(path_probs[0,:])
	for i in range(steps):
		alph = abs(path_probs[:,i]-0.001)
		axis = axarr[np.floor(i/subplot_dims[1]),i%subplot_dims[1]]
		green = np.zeros([all_states.shape[0],4]);blue = np.zeros([all_states.shape[0],4])
		blue[:,:3] = np.array([0,0,1]*all_states.shape[0]).reshape(all_states.shape[0],3) ; blue[:,3] = list(alph)
		green[:,:3] = np.array([0,1,0]*all_states.shape[0]).reshape(all_states.shape[0],3) ; green[:,3] = list(alph)
		axis.scatter(disc_model.target[0],disc_model.target[1],color="r",alpha = 0.0001)
		axis.scatter(all_states[:,0],all_states[:,1],color=blue)
		axis.scatter(all_states[:,2],all_states[:,3],color=green)
	fig.savefig(save_name+".png",dpi=80)
def learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = True):
	#initialise the lagrange multipliers to 1
	direc ="results/"
	fn.make_dir(direc)
	theta_1 = np.ones(len(expert1.w))*10
	theta_2 = np.ones(len(expert1.w))*10
	theta_3 = np.ones(len(expert1.w))*10
	D = 1.0
	disc = expert1.disc
	#----------------------------------------
	
	#test_states=initial_states
	a,s,f = expert1.feature_f.shape
	#experts
	exp1_policy,ignore,exp1_state_exp,exp1_all = inference(expert1,steps,initial_states)
	exp2_policy,ignore,exp2_state_exp,exp2_all = inference(expert2,steps,initial_states)

	exp1_feature_avg = np.dot(exp1_state_exp.reshape(s*a),expert1.feature_f.reshape(s*a,f))
	exp2_feature_avg = np.dot(exp2_state_exp.reshape(s*a),expert2.feature_f.reshape(s*a,f))

	expert1_value = eval_value(expert1.w,exp1_policy,expert1,test_states,steps)
	expert2_value = eval_value(expert2.w,exp2_policy,expert2,test_states,steps)
	expert_on_taboo = eval_value(expert2.w,exp1_policy,expert2,test_states,steps)


	z_stat = None

	#initiate results structure
	results = EmptyObject()
	results.value1 = []
	results.value2 = []
	results.policy_diff1 = []
	results.policy_diff2 = []
	results.expert1_value = expert1_value
	results.expert2_value = expert2_value
	results.test_value = expert_on_taboo
	rate = 0.6
	for i in range(iterations):
		apprentice_policy,z_stat,a_state_exp,a_all = inference(apprentice,steps,initial_states,z_states = z_stat)
		apprentice_feature_avg = np.dot(a_state_exp.reshape(s*a),apprentice.feature_f.reshape(s*a,f))
		difference_exp1 = exp1_feature_avg - apprentice_feature_avg
		difference_exp2 = exp2_feature_avg - apprentice_feature_avg
		print "First gradient",difference_exp1
		print "Second gradient",difference_exp2 
		#print "alpha minus",(-apprentice.zeta +D +theta_2)/theta_3
		#print "alpha plus",(apprentice.zeta +D +theta_1)/theta_3
		
		apprentice.w = apprentice.w + rate*difference_exp1
		
		if failure == True:
			zeta_prev =np.zeros(apprentice.zeta.shape) + apprentice.zeta
			#apprentice.zeta =apprentice.zeta - rate*(difference_exp2-(theta_1-theta_2+2*zeta_prev)/theta_3)
			update = np.zeros(len(difference_exp2))
			idx = np.where(difference_exp1*difference_exp2<0.00)[0]
			update[idx]=difference_exp2[idx]
			#update = difference_exp2+ 0.4*zeta_prev
			print "UPdate",update
			apprentice.zeta =apprentice.zeta - rate*(update)
		#print "w",apprentice.w
		#print "TESTS--------------------------"
		#print "alpha plus", (-zeta_prev-D+theta_2)/theta_3
		#print "alpha minus", (zeta_prev-D+theta_1)/theta_3
		#print "-------------------------------"

		#print "ZETA",apprentice.zeta
		#theta_1_temp = theta_1 -rate*((zeta_prev+D - theta_2)/theta_3)
		#theta_2_temp = theta_2 -rate*((-zeta_prev+D - theta_1)/theta_3)

		#print "RATE",((-zeta_prev-D - theta_1)/theta_3)
		#theta_1_temp = pin_to_zero(theta_1_temp)
		#theta_2_temp = pin_to_zero(theta_2_temp)
		#theta_3 = theta_3 + rate*(D*(theta_1+theta_2-D)/theta_3**2 + (zeta_prev*(-theta_2+theta_1+zeta_prev))/theta_3**2 - (theta_1*theta_2)/theta_3**2)
		#theta_3 = pin_to_zero(theta_3)

		#####################ALPHA CHECKING PART#####################
		#alpha_plus = (-apprentice.zeta-D+theta_2_temp)/theta_3
		#alpha_minus = (apprentice.zeta-D+theta_1_temp)/theta_3

		#print "_______________________________________"
		#print "JSUT OUTSIDE OF THE THINGY"
		#print theta_1_temp
		#print "-----"
		#print theta_2_temp
		#print alpha_plus
		#print alpha_minus
		#print "---------------------------------------"
		cond = False
		if cond ==True:
			flags = []
			matrix1 = np.array([[0.5,-0.5],[-0.5,0.5]])
			matrix2 = np.array([[0.5,0.5],[0.5,0.5]])
			for n,i in enumerate(alpha_plus):
				if i<0:
					update=np.dot(matrix1, np.array([apprentice.zeta[n]-zeta_prev[n],theta_2_temp[n]-theta_2[n]]))
					flags.append( True )
					apprentice.zeta[n] = update[0]
					theta_2_temp[n] = update[1]
				else:
					flags.append(False)
			alpha_minus = (apprentice.zeta-D+theta_1_temp)/theta_3
			for n,i in enumerate(alpha_minus):
				if i<0 and flags[n] == True:
					update=np.dot(matrix2, np.array([theta_1_temp[n]-theta_1[n],theta_2_temp[n]-theta_2[n]]))
					theta_1_temp[n] = update[0]
					theta_2_temp[n] = update[1]
				elif i<0 and flags[n]==False:
					update=np.dot(matrix2, np.array([apprentice.zeta[n]-zeta_prev[n],theta_1_temp[n]-theta_1[n]]))
					apprentice.zeta[n] = update[0]
					theta_1_temp[n] = update[1]

		#############################################################
		#alpha_plus = (-apprentice.zeta-D+theta_2)/theta_3
		#alpha_minus = (apprentice.zeta-D+theta_1_temp)/theta_3
		#print "CHECK IF POSSITIVEE---------------------------------------------------"
		#print alpha_plus
		#print alpha_minus
		apprentice.reward_f = apprentice.buildRewardFunction()
		#theta_1 = theta_1_temp
		#theta_2 = theta_2_temp
		#for n in range(len(theta_1_temp)):
		#	if theta_1_temp[n]>=0:
		#		theta_1[n]=theta_1_temp[n]
		#	if theta_2_temp[n]>=0:
		#		theta_2[n]=theta_2_temp[n]
		#theta_1 =theta_1_temp
		#theta_2 = theta_2_temp
		#print "Theta 1",theta_1
		#print "Theta 2",theta_2
		#print "Theta 3",theta_3
		
		value = eval_value(expert1.w,apprentice_policy,apprentice,test_states,steps) 
		value2 = eval_value(expert2.w,apprentice_policy,apprentice,test_states,steps) 
		print "failure",failure
		print "Iteration",i
		print "Aprentice on Expert" ,value
		print "Expert on expert",expert1_value
		print "Apprentice on Taboo",value2
		print "Taboo on Taboo",expert2_value
		print "Expert on Taboo",expert_on_taboo
		print "______________________________________"


		
		#if i ==0:
		#	plot_path(a_all,expert1.disc,[2,5],direc+"TALK_BEFORE_learning_2_path")
		#if i ==iterations-1:
	#		plot_path(a_all,expert1.disc,[2,5],direc+"TALK_after_learning_2_path")
		results.value1.append(value)
		results.value2.append(value2)
		results.policy_diff1.append(sum(sps.entropy(apprentice_policy,qk = exp1_policy)))
		results.policy_diff2.append(sum(sps.entropy(apprentice_policy,qk = exp2_policy)))
		#if value>expert1_value:
		#	print "converged in ",i
			#break
		rate = rate/1.1
		print "RATE",rate
	return results
		#### Result plots ####
def results_plot(results,name):
	f,axarr = plt.subplots(3,1,sharex=True)
	x = range(iterations)
	axarr[0].scatter(x,results.value1,color = "b",marker = '.',label= "apprentice policy expert rf")
	axarr[0].plot(x,[results.expert1_value]*iterations,color = "k",label = "expert value")
	axarr[0].set_ylabel("Value")
	axarr[0].set_xlim((0,iterations+40))
	axarr[0].legend(bbox_to_anchor=(1.1, 0.7,0.,-0.1),loc=1, prop={"size":11})

	axarr[1].set_ylabel("Value")
	axarr[1].set_xlim((0,iterations+40))
	axarr[1].scatter(x,results.value2,color = "r",marker = 'v',label= "apprentice policy non expert rf")
	axarr[1].plot(x,[results.expert2_value]*iterations,color = "m",label = "non expert value")
	axarr[1].plot(x,[results.test_value]*iterations,color = "g",label = "expert policy on non expert rf")
	axarr[1].legend(bbox_to_anchor=(1.1, 0.7,0.,-0.1),loc=1, prop={"size":11})
	fn.make_dir("results")
	axarr[2].plot(x,results.policy_diff1,color ="b",label = "policy difference with expert" )
	axarr[2].plot(x,results.policy_diff2,color = "r",label = "policy difference with non expert")
	axarr[2].legend(bbox_to_anchor=(1.1, 0.7,0.,-0.1),loc=1)
	axarr[2].set_ylabel("Sum Policy KL divergence")
	#f.set_size_inches(22.,17.)
	f.savefig(name+'.png',dpi=80)

def results_plot2(results1,results2,name):
	f,axarr = plt.subplots(1,1,sharex=False)
	iterations = len(results1.value1)
	x = range(iterations)

	axarr.scatter(x,results1.value1,color = "b",marker = '.',label= "Apprentice on Expert. Our Method")
	axarr.scatter(x,results2.value1,color = "r",marker = '.',label= "Apprentice on Expert. Original")
	axarr.plot(x,[results1.expert1_value]*iterations,color = "k",label = "Expert value",linewidth=2)
	axarr.set_ylabel("Value (Re)")
	axarr.set_xlim((0,iterations+5))
	axarr.legend(bbox_to_anchor=(1.0, 0.5,0.,0.0),loc=1, prop={"size":15})
	axarr.set_xlabel("iterations")
	f.savefig(name+'c.png',dpi=80)
	f,axarr = plt.subplots(1,1,sharex=False)
	axarr.set_ylabel("Value (Rt)")
	axarr.set_xlim((0,iterations+5))
	axarr.scatter(x,results1.value2,color = "b",marker = '.',label= "Apprentice on Taboo. Our Method")
	axarr.scatter(x,results2.value2,color = "r",marker = '.',label= "Apprentice on Taboo. Original")
	#axarr[1].plot(x,[results.expert2_value]*iterations,color = "m",label = "non expert value")
	axarr.plot(x,[results1.test_value]*iterations,color = "g",label = "Expert on Taboo",linewidth=2)
	axarr.legend(bbox_to_anchor=(1.0, 1.0,0.,0.0),loc=1, prop={"size":15})
	axarr.set_xlabel("iterations")
	# axarr[2].plot(x,results1.policy_diff1,color ="b",label = "policy difference expert failure" )
	# axarr[2].scatter(x,results1.policy_diff2,color = "b",marker='v',label = "policy difference non expert failure")
	# axarr[2].plot(x,results2.policy_diff1,color ="r",label = "policy difference expert original" )
	# axarr[2].scatter(x,results2.policy_diff2,color = "r",marker = 'v',label = "policy difference non expert original")

	# axarr[2].legend(bbox_to_anchor=(1.1, 0.7,0.,-0.1),loc=1,prop={"size":9})
	# axarr[2].set_ylabel("Sum Policy KL divergence")
	#f.set_size_inches(22.,17.)
	f.savefig(name+'d.png',dpi=80)


if __name__ == "__main__":
	def test_normal_learn():
		disc = DiscModel(target = [4,4],boundaries = [4,4])
		expert = Model(disc,"avoid_reach", load_saved = True)
		initial_states = [disc.quantityToState([0,0,1,3,1])]
		apprentice = Model(disc,"uniform", load_saved = True)
		iterations = 400
		steps = 10
		initial_states = [disc.quantityToState([0,0,1,3,1]),disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,1,2,1])]
		print expert.w
		#initial_states = range(disc.tot_states)
		learn_normal(expert,apprentice,iterations,steps,initial_states)

	#test_normal_learn()

	def run_case1():
		fn.make_dir("results/test1")
		state_dictionary = ({"a_clean":[0,1],"b_clean":[0,1],"a_shaped":[0,1],"b_shaped":[0,1],
						"a_painted":[0,1],"b_painted":[0,1],"a_drilled":[0,1],"b_drilled":[0,1],
						"joined":[0,1]})
		actions = (["paint_a","paint_b","shape_a","shape_b","drill_a",
					"drill_b","wash_a","wash_b","bolt","glue"])
		disc = DiscModel(state_dictionary,actions)
		expert1 = Model(disc,"original", load_saved = False)
		expert2 = Model(disc,"bad", load_saved = False)
		print expert2.w
		print expert1.w
		apprentice = Model(disc,"dual_reward", load_saved = False)
		
		iterations = 200
		steps = 40
		#
		initial_states = [disc.quantityToState(np.zeros(9)),2,3,4,5,6,7,8]
		test_states = initial_states
		
		results1 = learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = True)
		#apprentice = Model(disc,"dual_reward", load_saved = True)
		#results2 = learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = False)
		#results 1 is faulure on the legends. results 2 is the normal
		#results_plot2(results1,results2,"results/workshop/results_normal_allstates_case1")

		#apprentice = Model(disc,"dual_reward", load_saved = True)
		#initial_states = [disc.quantityToState([0,0,1,3,2]),disc.quantityToState([0,0,3,1,4]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,3,1]),disc.quantityToState([0,0,2,2,4]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]
		#initial_states = [disc.quantityToState([0,0,1,3,2])];test_states = [disc.quantityToState([0,0,2,3,3])]
		#results2 = learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = False)
		#apprentice = Model(disc,"dual_reward", load_saved = True)
		#results1 = learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = True)
		#results 1 is faulure on the legends. results 2 is the normal
		#fn.pickle_saver(results1,"results/workshop/results1")
		#fn.pickle_saver(results2,"results/workshop/results2")
		#results_plot2(results1,results2,"results/workshop/results_normal_partial_case1")

	#results1 = fn.pickle_loader("results/workshop/results1")
	#results2 = fn.pickle_loader("results/workshop/results2")
	#results_plot2(results1,results2,"results/workshop/results_normal_partial_case1")
	#results_plot2(results1,results2,"results/workshop/results_normal_partial_case1")
	def run_case2():
		direc = "results/Tuesday07062015/factory"
		fn.make_dir(direc)
		results_array = []
		runs = 100 # number of test and train states
		iterations_per_run = 20
		steps = 15

		state_dictionary = ({"a_clean":[0,1],"b_clean":[0,1],"a_shaped":[0,1],"b_shaped":[0,1],
						"a_painted":[0,1],"b_painted":[0,1],"a_drilled":[0,1],"b_drilled":[0,1],
						"joined":[0,1]})
		actions = (["paint_a","paint_b","shape_a","shape_b","drill_a",
					"drill_b","wash_a","wash_b","bolt","glue"])
		disc = DiscModel(state_dictionary,actions)
		expert1 = Model(disc,"original", load_saved = False)
		expert2 = Model(disc,"bad", load_saved = True)

		for i in range(runs):
			apprentice = Model(disc,"dual_reward", load_saved = True)
			initial_states = np.random.randint(0,disc.tot_states,2)
			test_states = np.random.randint(0,disc.tot_states,2)
			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = True)
			apprentice = Model(disc,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = False)
			results_array.append([results_failure,results_normal])

		fn.pickle_saver(results_array,direc+"/results_factory.pkl")
		all_results = fn.pickle_loader(direc+"/results_factory.pkl")
		print all_results

	run_case2()






