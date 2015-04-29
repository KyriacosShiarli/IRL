import sys
import math
import numpy as np
import random as rd
import time

class GridWorld():
  def __init__(self,layout,obstacle=6,goal=15):
    self.num_states = layout[0]*layout[1]
    self.layout = layout
    self.obstacle = obstacle
    self.goal = goal
    self.actions =['north','south','east','west','stay']
    self.build_reward_function()
    self.build_transition_function()
  def build_reward_function(self,weights= np.array([-0.2,-0.3])):
    #weights = weights/np.linalg.norm(weights)
    self.reward_f = np.zeros(self.layout[0]*self.layout[1])
    o_coord = state_to_coordinate(self.obstacle,self.layout)
    g_coord = state_to_coordinate(self.goal,self.layout)
    for i in range(self.layout[1]*self.layout[0]):
      a_coord = state_to_coordinate(i,self.layout)
      features = calc_features(a_coord,o_coord,g_coord,math.sqrt(self.layout[0]**2+self.layout[1]**2))
      self.reward_f[i] = sum(weights*features)    
    print "Reward Function built"
  def query_transition_function(self,state,action):
    #given a state and action returns a distribution over next states
    coord = state_to_coordinate(state,self.layout)
    out = np.zeros(self.num_states)
    new_coord = [0,0]
    if action == "north":
      if coord[1]==0:
        new_coord = coord
      else:
        new_coord = np.array([coord[0],coord[1]-1])
    elif action == "south":
      if coord[1]==self.layout[1]-1:
        new_coord = coord
      else:
        new_coord = np.array([coord[0],coord[1]+1])
    elif action ==  "east":
      if coord[0]==self.layout[0]-1:
        new_coord = coord
      else:
        new_coord = np.array([coord[0]+1,coord[1]])
    elif action ==  "west":
      if coord[1]==0:
        new_coord = coord
      else:
        new_coord = np.array([coord[0]-1,coord[1]])
    elif action ==  "stay":
      new_coord = coord
    new_state = coordinate_to_state(new_coord,self.layout)
    out[new_state] = 1
    if action == "all":
      actions = ["north","south","east","west","stay"]
      out = np.zeros([len(actions),self.num_states])
      for (i, act) in  enumerate(actions):
        out[i,:] = self.query_transition_function(state,act)
    return out
  def build_transition_function(self):
    transition_f = np.zeros([len(self.actions),self.num_states,self.num_states])
    for i in range(self.num_states):
      transition_f[:,i,:] = self.query_transition_function(i,"all")
    self.transition_f = transition_f
    print "Transition Function Built"

def state_to_coordinate(state,layout):
  assert state < layout[0]*layout[1]
  coordinate = np.array([math.fmod(state,layout[0]),math.floor(state/layout[0])])
  return coordinate

def coordinate_to_state(coord,layout):
  state = coord[1]*layout[0] + coord[0]
  return state

def calc_features(agent_coords,obstacle_coords,goal_coords,max_dist):
  feature1 = (max_dist - np.linalg.norm(agent_coords-obstacle_coords))/(max_dist/6)
  feature2 = (np.linalg.norm(agent_coords-goal_coords))/(max_dist/6)
  return np.array([feature1,feature2])


if __name__ == "__main__":
  obstacle = 50
  goal = 99
  layout = [70,70]
  env = GridWorld(layout,obstacle,goal)
  w_init = [-5,-10]
  w_init = w_init/np.linalg.norm(w_init)
  env.build_reward_function(w_init)
  freq,t = forward_backard(env.transition_f,env.reward_f,0,goal,99)


  o_coord = state_to_coordinate(obstacle,layout)
  g_coord = state_to_coordinate(goal,layout)

  w = [-26,-1]
  grad_prev = 0
  for k in range(200):
    print "INITIAL",w_init
    w = w/np.linalg.norm(w)
    print "Iterarion ----------->", k
    env.build_reward_function(w)
    freq2,t = forward_backard(env.transition_f,env.reward_f,0,goal,99)
    cumulative1 = np.zeros(2)
    cumulative2 = np.zeros(2)
    for i,j in enumerate(freq):
     coord = state_to_coordinate(i,layout)
     feat = calc_features(coord,o_coord,g_coord,math.sqrt(layout[0]**2+layout[1]**2))
     cumulative1+= j * feat
     cumulative2+= freq2[i] * feat 

    print cumulative1,cumulative2
    print cumulative1-cumulative2
    gamma = 0.01 * k**2
    w = w * np.exp(-gamma*(cumulative1-cumulative2))
    w = w/np.linalg.norm(w)
    grad_prev = cumulative1-cumulative2
    print "W = ",w


