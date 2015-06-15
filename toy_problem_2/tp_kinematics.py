import numpy as np
import math
# AXIS INFORMATION:
			
#				.-----> x
#				|
#				|
#				|
#				| y
#
#				actions: 0 = stay
#						 1 = x +1
#						 2 = x -1
#						 3 = y +1
#						 4 = y -1

def kinematics_builder(state,action):
	def paint_a(state):
		if state[0] == 1:
			out1 = list(state)
			out1[4]=1
			out2 = list(state);out2[0]=0
			return [[out1,0.75],[out1,0.25],[state,0.05]]
		else:
			return [[state,1]]
	def paint_b(state):
		if state[1] == 1:
			out1 = list(state);out1[5]=1
			out2 = list(state);out2[1]=0
			return [[out1,0.75],[out2,0.25],[state,0.05]]
		else:
			return [[state,1]]
	def shape_a(state):
		if state[-1] == 0:
			out1 = list(state);out1[4]=0;out1[2]=1
			out2 = list(state);out2[4]=0;out2[0]=0;out2[2]=0;out2[6]=0
			out3 = list(state);out3[4]=0
			return [[out1,0.8],[out2,0.1],[out3,0.1]]
		else:
			out1=list(state);out1[4]=0;out1[2]=1
			return [[out1,1]]
	def shape_b(state):
		if state[-1] == 0:
			out1 = list(state);out1[5]=0;out1[3]=1
			out2 = list(state);out2[5]=0;out2[1]=0;out2[3]=0;out2[7]=0
			out3 = list(state);out3[5]=0
			return [[out1,0.8],[out2,0.1],[out3,0.1]]
		else:
			out1=list(state);out1[5]=0;out1[3]=1
			return [[out1,1]]
	def drill_a(state):
		if state[-1] == 0:
			out1 = list(state);out1[6]=1
			out2 = list(state);
			return [[out1,0.9],[out2,0.1]]
		else:
			return [[state,1]] 
	def drill_b(state):
		if state[-1] == 0:
			out1 = list(state);out1[7]=1
			out2 = list(state);
			return [[out1,0.9],[out2,0.1]]
		else:
			return [[state,1]]
	def wash_a(state):
		out1 = list(state);out1[0]=1
		return [[out1,0.9],[state,0.1]]
	def wash_b(state):
		out1 = list(state);out1[1]=1
		return [[out1,0.9],[state,0.1]]
	def bolt(state):
		if state[2]==1 and state[3]==1 and state[6]==1 and state[7]==1:
			out1 = list(state);out1[8] = 1
			return[[out1,0.8],[state,0.2]]
		else: return [[state,1]]
	def glue(state):
		if state[2]==1 and state[3]==1:
			out1 = list(state);out1[0]=0;out1[1]=0;out1[-1]=1
			out2 = list(state);out2[-1]=1
			out3 = list(state); out3[0]=0;out3[1]=0
			out4 = list(state);
			return [[out1,0.35],[out2,0.35],[out3,0.15],[out4,0.15]]
		elif state[2]==0:
			out1 = list(state);out1[0]=0;out1[1]=0
			out2 = list(state)
			return [[out1,0.5],[out2,0.5]]
		elif state[2]==1 and state[3]==1:
			out1 = list(state);out1[0]=0;out1[1]=0
			out2 = list(state)
			return [[out1,0.5],[out2,0.5]]
		else:return [[state,1]]

	#print action
	return vars()[action](state)

if __name__=="__main__":
	#out = staticGroup_with_target([0.5,0.5,0.7,0.9],[0.1,0.2])
	state = [0,15,3,3,4]
	action = 0
	for i in range(25):
		state = toy_kinematics_gridworld(state,action,[15,15])
		print state