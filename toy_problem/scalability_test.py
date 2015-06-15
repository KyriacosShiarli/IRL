import tp_forwardBackward as fb
from tp_Model import *
from tp_discretisationmodel import *
import tp_functions as fn
from tp_evaluation import *
import scipy.stats as sps
from tp_data_structures import EmptyObject
time = []
dims = range(5,30,5)
for i in dims:
	disc = DiscModel(target = [i,i],boundaries = [i,i])
	expert1 = Model(disc,"avoid_reach", load_saved = False)
	p,lp,z,times = fb.timed_backward(expert1.transition,expert1.reward_f,conv=5,discount = 0.9,z_states = None)
	times.append(disc.tot_states)
	time.append(times)
	print times 
fn.pickle_saver(times,"scalability/times")

