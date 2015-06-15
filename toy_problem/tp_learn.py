import tp_forwardBackward as fb
from tp_Model import *
from tp_discretisationmodel import *
import tp_functions as fn
from tp_evaluation import *
import scipy.stats as sps
from tp_data_structures import EmptyObject
from tp_RFeatures import toy_problem_simple,toy_problem_squared

def pin_to_zero(vector_to_pin):
	vector_to_pin = np.array(vector_to_pin)
	where = [i for i in range(len(vector_to_pin)) if vector_to_pin[i] < 0]
	vector_to_pin[where]= 0
	return vector_to_pin
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
def learn_from_failure(expert1,expert2,apprentice,iterations,steps,initial_states,test_states,failure = True,typ=None):
	#initialise the lagrange multipliers to 1
	print "INITIALISED LEARNING. LEARNING FROM FAILURE = ",failure,typ
	direc ="results/"
	fn.make_dir(direc)
	D_max=3.0;D_min=-3.0
	D=3.0
	theta_1 = np.ones(len(expert1.w))*0
	theta_2 = np.ones(len(expert1.w))*2*D
	theta_3 = np.ones(len(expert1.w))*5
	C = 30.0
	disc = expert1.disc
	apprentice.zeta = -D

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

	plot_path(exp1_all,expert1.disc,[6,5],direc+"TALK_expert1_path")
	plot_path(exp2_all,expert1.disc,[6,5],direc+"TALK_expert2_path")

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
	rate = 0.3

	for i in range(iterations):
		apprentice_policy,z_stat,a_state_exp,a_all = inference(apprentice,steps,initial_states,z_states = z_stat)
		print z_stat
		apprentice_feature_avg = np.dot(a_state_exp.reshape(s*a),apprentice.feature_f.reshape(s*a,f))
		difference_exp1 = exp1_feature_avg - apprentice_feature_avg
		difference_exp2 = exp2_feature_avg - apprentice_feature_avg
		#print "First gradient",difference_exp1
		#print "Second gradient",difference_exp2 
		#print "alpha minus",(-apprentice.zeta +D +theta_2)/theta_3
		#print "alpha plus",(apprentice.zeta +D +theta_1)/theta_3

		if failure== True:
			apprentice.w = fn.pin_to_threshold(apprentice.w + rate*difference_exp1,C,-C)
			if i >7:

				idx = np.where(difference_exp1*difference_exp2>0.00)[0]
				if typ =="Normal":
					zeta_prev = apprentice.zeta
					#print "UP",difference_exp2-(theta_1-theta_2+2*zeta_prev)/theta_3
					apprentice.zeta =fn.pin_to_threshold( apprentice.zeta + rate*(-difference_exp2-(theta_1-theta_2+2*zeta_prev)/theta_3),D,-D)
					#apprentice.zeta = -D+theta_2
					theta_1_temp = fn.pin_to_threshold( theta_1 -rate*((zeta_prev+D - theta_2)/theta_3),2*D,0)
					theta_2_temp =fn.pin_to_threshold( theta_2 -rate*((-zeta_prev+D - theta_1)/theta_3),2*D,0)
					theta_3 = theta_3 + rate*(D*(theta_1+theta_2-D)/theta_3**2 + (zeta_prev*(-theta_2+theta_1+zeta_prev))/theta_3**2 - (theta_1*theta_2)/theta_3**2)
					theta_1 =theta_1_temp
					theta_2 = theta_2_temp
					#print theta_1
					#print theta_2
					#print "THETA@",theta_3	
					#apprentice.zeta[idx] = 0
				#print "ZETA",apprentice.zeta
		else:
			apprentice.w = apprentice.w + rate*difference_exp1
		#else: zeta_prev = apprentice.w
		#print "w",apprentice.w
		#print "TESTS--------------------------"
		#print "alpha plus", (-zeta_prev-D+theta_2)/theta_3
		#print "alpha minus", (zeta_prev-D+theta_1)/theta_3
		#print "-------------------------------"

		#print "ZETA",apprentice.zeta
		#theta_1_temp = theta_1 -rate*((zeta_prev+D - theta_2)/theta_3)
		#theta_2_temp = theta_2 -rate*((-zeta_prev+D - theta_1)/theta_3)

		#theta_1_temp = pin_to_zero(theta_1_temp)
		#theta_2_temp = pin_to_zero(theta_2_temp)
		#theta_3 = theta_3 + rate*(D*(theta_1+theta_2-D)/theta_3**2 + (zeta_prev*(-theta_2+theta_1+zeta_prev))/theta_3**2 - (theta_1*theta_2)/theta_3**2)
		#theta_3 = pin_to_zero(theta_3)

		#####################ALPHA CHECKING PART#####################
		#alpha_plus = (-apprentice.zeta-D+theta_2_temp)/theta_3
		#alpha_minus = (apprentice.zeta-D+theta_1_temp)/theta_3

		#print "JSUT OUTSIDE OF THE THINGY"

		#print theta_1_temp

		#print theta_2_temp
		#print alpha_plus
		#print alpha_minus
		
		# cond = False
		# if cond==True:
		# 	flags = []
		# 	matrix1 = np.array([[0.5,-0.5],[-0.5,0.5]])
		# 	matrix2 = np.array([[0.5,0.5],[0.5,0.5]])
		# 	for n,j in enumerate(alpha_plus):
		# 		if j<0:
		# 			update=np.dot(matrix1, np.array([apprentice.zeta[n]-zeta_prev[n],theta_2_temp[n]-theta_2[n]]))
		# 			flags.append( True )
		# 			apprentice.zeta[n] = update[0]
		# 			theta_2_temp[n] = update[1]
		# 		else:
		# 			flags.append(False)
		# 	alpha_minus = (apprentice.zeta-D+theta_1_temp)/theta_3
		# 	for n,j in enumerate(alpha_minus):
		# 		if j<0 and flags[n] == True:
		# 			update=np.dot(matrix2, np.array([theta_1_temp[n]-theta_1[n],theta_2_temp[n]-theta_2[n]]))
		# 			theta_1_temp[n] = update[0]
		# 			theta_2_temp[n] = update[1]
		# 		elif j<0 and flags[n]==False:
		# 			update=np.dot(matrix2, np.array([apprentice.zeta[n]-zeta_prev[n],theta_1_temp[n]-theta_1[n]]))
		# 			apprentice.zeta[n] = update[0]
		# 			theta_1_temp[n] = update[1]

			#############################################################
		#	alpha_plus = (-apprentice.zeta-D+theta_2)/theta_3
		#	alpha_minus = (apprentice.zeta-D+theta_1_temp)/theta_3
		#print "CHECK IF POSSITIVEE---------------------------------------------------"
		#print alpha_plus
		#print alpha_minus
		apprentice.reward_f = apprentice.buildRewardFunction()
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

		
		if i ==0:
			plot_path(a_all,expert1.disc,[6,5],direc+"TALK_BEFORE_learning_2_path")
		if i ==iterations-1:
			plot_path(a_all,expert1.disc,[6,5],direc+"TALK_after_learning_2_path")
		results.value1.append(value)
		results.value2.append(value2)
		results.policy_diff1.append(sum(sps.entropy(apprentice_policy,qk = exp1_policy)))
		results.policy_diff2.append(sum(sps.entropy(apprentice_policy,qk = exp2_policy)))
		#rate = rate/1.03
	return results

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

	def experiment(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature",iterations_per_run=50,steps=15,runs=10):
		direc = "results/Tuesday07062015/test_margin_algorithm"
		#initial_states = [disc.quantityToState([0,0,1,2,2]),disc.quantityToState([0,0,3,4,1]),disc.quantityToState([0,1,2,2,2]),disc.quantityToState([0,0,3,2,1])]
		#test_states =[disc.quantityToState([0,0,2,2,1]),disc.quantityToState([0,0,2,4,2]),disc.quantityToState([0,0,3,1,3]),disc.quantityToState([0,0,3,2,1])]

		fn.make_dir("results/Tuesday07062015/test_margin_algorithm")
		results_array = []
		disc = DiscModel(target = [4,4],boundaries = [4,4],feature = toy_problem_simple)
		runs = 10 # number of test and train states
		expert2 = Model(disc,"obstacle2", load_saved = False)
		expert1 = Model(disc,"avoid_reach", load_saved = True)
		iterations_per_run = 50
		steps = 15
		for i in range(runs):
			apprentice = Model(disc,"dual_reward", load_saved = True)
			initial_states = np.random.randint(0,disc.tot_states,5)
			test_states = np.random.randint(0,disc.tot_states,5)

			results_failure = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = True,typ="Normal")
			apprentice = Model(disc,"dual_reward", load_saved = True)
			results_normal = learn_from_failure(expert1,expert2,apprentice,iterations_per_run,steps,initial_states,test_states,failure = False)
			results_array.append([results_failure,results_normal])
		fn.pickle_saver(results_array,direc+"/"+name+".pkl")
	#direc = "results/Tuesday07062015/test_margin_algorithm"
	#all_results = fn.pickle_loader(direc+"/results.pkl")
	def results_plots():
		direc = "results/Tuesday07062015/test_margin_algorithm"
		all_results = fn.pickle_loader(direc+"/results_feature_squared.pkl")
		print len(all_results)
		width = 2
		spacing = 0.3
		x = width+spacing+width/2
		f = plt.figure()
		f,axarr = plt.subplots(2,sharex=False);ax = axarr[0];ax2=axarr[1]
		width = 0.15

		avg_diff_expert_failure = [];avg_diff_expert_normal = [];avg_diff_taboo_failure=[];avg_diff_taboo_normal=[]
		for result in all_results:
			print result[0].value1[-1], result[1].value1[-1]
			avg_diff_expert_failure.append(result[0].value1[-1]-result[0].expert1_value)
			avg_diff_expert_normal.append(result[1].value1[-1]-result[1].expert1_value)
			avg_diff_taboo_failure.append(result[0].value2[-1]-result[0].test_value)
			avg_diff_taboo_normal.append(result[1].value2[-1]-result[1].test_value)

		print "Diff Expert failure",np.mean(avg_diff_expert_failure),np.std(avg_diff_expert_failure)
		print "Diff Expert Normal",np.mean(avg_diff_expert_normal),np.std(avg_diff_expert_normal)
		print "Diff Taboo Failure",np.mean(avg_diff_taboo_failure),np.std(avg_diff_taboo_failure)
		print "Diff Taboo Normal",np.mean(avg_diff_taboo_normal),np.std(avg_diff_taboo_normal)
		results_plot2(all_results[1][0],all_results[1][1],direc+"/plots.pkl")
	results_plots()
	#print all_results
	experiment(expert_feature = toy_problem_squared,apprentice_feature = toy_problem_squared,name = "eucledian_feature")
	experiment(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_simple,name = "simple_feature")
	experiment(expert_feature = toy_problem_squared,apprentice_feature = toy_problem_simple,name = "cross_feature_expert_euclid")
	experiment(expert_feature = toy_problem_simple,apprentice_feature = toy_problem_squared,name = "cross_feature_expert_xy")
	
		








