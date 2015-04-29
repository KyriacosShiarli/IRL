import sys
import numpy as np
import math
import random as rd
import time
from tp_functions import sum_chunks
def smart_backward(transition,reward_f,goal,conv=0.1,z_states = None):
    num_actions = transition.tot_actions;num_states = transition.tot_states
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = -np.ones(num_states)*1.0e5
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Unormalised measure calculations done in log space to prevent nans
    print "Backward"
    delta = 10
    policy = np.zeros([num_actions,num_states])
    while delta > conv or math.isnan(delta) == True:
      prev = np.zeros((num_actions,num_states))
      prev += policy
      for i in range(num_states):
        for j in range(num_actions):
          m = np.amax(z_states)
          keys = map(int,transition.backward[j + i*num_actions].keys())
          val = transition.backward[j + i*num_actions].values()
          z_actions[j,i] =m+np.log(np.sum(np.array(val)*np.exp(z_states[keys]-m))) + reward_f[i]
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      #Action Probability Computation - - - - - - - - - - - - - - - -
      policy= np.exp(z_actions-z_states)
      delta = sum(sum(np.absolute(prev-policy)))
    print "Finished Backward. Converged at: %s"%delta
    return policy,z_states

def forward(policy,transition,start,time_steps):
    print "START of FORWARD ------------------->"
    num_actions = transition.tot_actions;num_states = transition.tot_states
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states_actions = np.zeros([num_actions,num_states,time_steps])
    for i in start: dt_states[i,0]+=1 
    dt_states[:,0] /=len(start)
    print "HEERE", dt_states[:,0]
    for i in range(time_steps):
      for j in range(num_states):
        tr = transition.dense_forward[j]
        if i != time_steps-1:
          dt_states[j,i+1] = np.sum(dt_states[map(int,tr[1,:]),i] *policy[map(int,tr[0,:]),map(int,tr[1,:])] * tr[2,:]) 
        dt_states_actions[:,j,i] = dt_states[j,i]*policy[:,j]
    state_action_freq = np.sum(dt_states_actions,axis=2)        
    state_freq = np.sum(dt_states,axis = 1)
    print "END of Forward"
    return state_freq,state_action_freq,dt_states

def forward_approx(policy,model,start,steps,repetitions = 1000):
  state_action_freq = np.zeros([model.disc.tot_actions,model.disc.tot_states])
  print "START", start
  print "DIVISION",len(start)*repetitions
  for i in start:
    init = model.disc.stateToQuantity(i)
    for j in range(repetitions):
      state_action_freq+= trajectory_from_policy(model,policy,init,steps,counts = True)
      tock = time.clock()
  print "SHOULD SUM TO 30",np.sum(np.sum(state_action_freq))/(len(start)*repetitions)
  return state_action_freq/(len(start)*repetitions)

def forward_uni(policy,transition,time_steps):

    print "STARTING FORWARD ------------------->"

    num_actions = transition.tot_actions;num_states = transition.tot_states
    start = range(num_states)
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states_actions = np.zeros([num_actions,num_states,time_steps])
    for i in start: dt_states[i,0]+=1 
    dt_states[:,0] /=len(start)
    for i in range(time_steps):
      for j in range(num_states):
        tr = transition.dense_forward[j]
        if i != time_steps-1:
          dt_states[j,i+1] = np.sum(dt_states[map(int,tr[1,:]),i] *policy[map(int,tr[0,:]),map(int,tr[1,:])] * tr[2,:]) 
        dt_states_actions[:,j,i] = dt_states[j,i]*policy[:,j]
    state_action_freq = np.sum(dt_states_actions,axis=2)        
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq,state_action_freq

    
def caus_ent_backward(transition,reward_f,goal,steps,conv=5,z_states = None):
    num_actions = transition.tot_actions;num_states = transition.tot_states
    if reward_f.shape[0] ==num_actions:
      state_action = True
    else: state_action =False
    gamma = 0.99
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = np.zeros(num_states)
      #z_states[goal] = 0
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Caus Ent Backward"
    count = 0
    delta = 0
    #reward_f[:,goal]/=1.1
    for j in range(steps):
      prev = np.zeros(z_states.shape)
      prev += z_states
      for i in range(num_states):
        tr = transition.dense_backward[i]
        ch = transition.chunks_backward[i]
        out = gamma*np.array(sum_chunks(tr[2,:]*z_states[map(int,tr[1,:])],ch))
        z_actions[:,i] = out +reward_f[:,i]
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      count+=1
      #Action Probability Computation - - - - - - - - - - - - - - - -
      delta = np.sum(np.sum(np.absolute(prev-z_states)))
      #delta +=1 
      #print "DElta cause",delta,delta2
      if j==steps-1:
        z_actions = z_actions
        m = np.amax(z_actions)
        z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
        policy= np.exp(z_actions-z_states)
    return policy,np.log(policy),z_states
'''
if __name__=="__main__":
  from discretisationmodel import *
  from Model import Model
  from discretisationmodel import 
  disc_model = DiscModel()
  w = None
  model = Model(disc_model,w,learn = False)
  p,lp,zs = caus_ent_backward(model.transition,model.reward_f,1,1,6,z_states=None)
  p2,lp2 = caus_ent_backward_test(model.transition,model.reward_f,1,1,6,z_states=None)
  print "FIRST DIFFERENCE", p-p2
  state_frequencies,sa= forward(p,model.transition,[1,2,3,4,5,6],5)
  state_frequencies_t,sa_t= forward_test(p,model.transition,[1,2,3,4,5,6],5)
'''
if __name__ == "__main__":
  from tp_discretisationmodel import *
  import matplotlib.pyplot as plt
  from tp_Model import *

  def test_toy_problem_inference():
    d = DiscModel()
    m = Model(d,load_saved = True)
    goal = 0
    steps = 30
    p,np,z=caus_ent_backward(m.transition,m.reward_f,goal,steps,conv=100,z_states = None)
    state_freq,state_action_freq = forward(p,m.transition,[1,2,3,4],30)
    print p.shape
    print state_freq
    #print "Difference", sa_t - sa
  test_toy_problem_inference()
  def test_forward_approximation():
    #load test policy from datasets
    test_policy = pickle_loader('TESTS/data/test_policy.pkl')
    for i in test_policy:print i
    w = None
    d = DiscModel()
    model = Model(d,w,learn = True)
    w = model.w 
    examples = load_all(30)
    start = [model.disc.quantityToState(te.states[0,:]) for te in examples]
    start = start[:1]
    tick = time.clock()
    sf,saf = forward(test_policy,model.transition,start,2)
    tock = time.clock()
    print "Full Time", tock-tick
    for i in range(1000,10000,1000):
      tick = time.clock()
      sa = forward_approx(test_policy,model,start,2,repetitions = i)
      tock = time.clock()
      print "Aproximation time", tock-tick
      print np.sum(np.sum(saf))
      print "Absolute difference of repetitions ",i, np.sum(np.sum(np.absolute(sa-saf)))

