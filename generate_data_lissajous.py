# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (260, 200, -1, -1),
#  'transitions': []}
### end of header
import time
import math
import random
import cPickle as pickle

from numpy import sin,pi,linspace
from pylab import plot,show,subplot

from pyutils.matrix import *

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    result = generate_data()

def execute_trajectory(pos_x, pos_y, pos_z, num_pt):

    idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([0, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0. ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)
    pos = cartToSpher(pos_x,pos_y,pos_z)
    pos_cart = [pos_x,pos_y,pos_z]
    # pos_x = 1.0 pos_y = -0.1 pos_z = 1.0
    goal_pos_prev = cartToSpher(pos_x,pos_y,pos_z) 
    goal_pos_prev_cart = [pos_x,pos_y,pos_z]
    
    app.rave.remove_coords("*")
    app.rave.clear_graphic_handles("*")
    q_dict = app.rave.get_config_dict()    
    #tr = app.rave.get_link_frames(["rightarm1"])["rightarm1"]
    tr = app.rave.get_manip_frame("right_arm")
    #app.rave.add_coord("tr", tr)
    offset = txyz(pos_x,pos_y,pos_z)
    center_frame = dot(offset, tr)
    center_frame[0:3,0:3] = dot(roty(pi/2.), rotz(pi/2))[0:3,0:3]
    #app.rave.add_coord("goal", center_frame)
    #print q_dict["right_arm"] 
    
    q_dict = app.find_ik_rotational(center_frame, 0)
    if q_dict is None:
        print "center frame unreachable!"
        return
    app.rave.set_config_dict(q_dict)      
    points = []
    colors = []
    
    motor_pos_prev = q_dict["right_arm"]
    motor_pos_center = q_dict["right_arm"]
    trial = []
    
    #lissajous function
    l = app.create_lissajous_positions(1, 8, num_points=num_pt, scale=0.1, do_plot=False)
    mf = app.rave.get_manip_frame("right_arm")
    for i, (x, y) in enumerate(zip(l[0], l[1])):        
        #generate trajectory
        #app.rave.add_coord("lissajous%d" % i, dot(txyz(*mf[0:3,3]), txyz(x, y, 0.)), "small" )
        #app.rave.add_coord("lissajous%d" % i, dot(mf, txyz(x, y, 0.)))
        next_pos = dot(mf, txyz(x ,y, 0.))
        #next_pos = dot(txyz(*mf[0:3,3]), txyz(x, y, 0.))
        #print next_pos
        #print next_pos[0:3,3]
        
        #execute trajectory
        #do InvKin
        tr1 = center_frame
        #offset1 = txyz(next_pos[0:3,3][0] - pos_x, next_pos[0:3,3][1] - pos_y, next_pos[0:3,3][2] - pos_z)
        #goal1 = dot(offset1, tr1)
        points.append(next_pos[0:3,3])
        
        initial = app.rave.get_config_dict()
        #app.rave.add_coord("goal", next_pos)
        #q_dict = app.find_ik_numerically(goal1, i)
        q_dict = app.find_ik_rotational(next_pos, i)
        if q_dict is None:
            colors.append((1, 0, 0))
            app.rave.draw_line("line", points, colors, 4)
            return None
        #qd = q_dict["right_arm"] - initial["right_arm"] 
        #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
        #    print "re-configuration not allowed"
        #    return  
          
        colors.append((0, 1, 0))
        app.rave.draw_line("line", points, colors, 4)
        app.rave.set_config_dict(q_dict)   
        reached_frame = app.rave.get_manip_frame("right_arm")
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        #sleep(0.2) 

        #prepare data
        time= i
        
        goal_pos = cartToSpher(next_pos[0:3,3][0], next_pos[0:3,3][1], next_pos[0:3,3][2])
        goal_delta = [goal_pos[0] - goal_pos_prev[0], goal_pos[1] - goal_pos_prev[1], goal_pos[2] - goal_pos_prev[2]]
        goal_pos_prev = goal_pos
        
        # for cartesian
        goal_pos_cart = [next_pos[0:3,3][0], next_pos[0:3,3][1], next_pos[0:3,3][2] ]
        goal_delta_cart = [goal_pos_cart[0] - goal_pos_prev_cart[0], goal_pos_cart[1] - goal_pos_prev_cart[1], goal_pos_cart[2] - goal_pos_prev_cart[2]]
        goal_pos_prev_cart = goal_pos_cart
        
        motor_pos = q_dict["right_arm"]
        motor_pos_delta = [motor_pos[0] - motor_pos_prev[0], motor_pos[1] - motor_pos_prev[1], motor_pos[2] - motor_pos_prev[2], motor_pos[3] - motor_pos_prev[3], motor_pos[4] - motor_pos_prev[4], motor_pos[5] - motor_pos_prev[5], motor_pos[6] - motor_pos_prev[6] ]
        motor_pos_prev = motor_pos
        
        trial.append([time, pos, goal_delta, goal_pos, motor_pos_center, motor_pos_delta, motor_pos, pos_cart, goal_delta_cart, goal_pos_cart ])
        #print "data:" , data        
        #print data
        #time.sleep(1)
        #write data


    return trial

# for equation: http://keisan.casio.com/exec/system/1359533867    
def cartToSpher(x,y,z):
    polar = [] 
    r = math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))
    # in radians
    theta = math.atan(y/x)
    phi = math.atan( (math.sqrt(math.pow(x,2) + math.pow(y,2) ) )/z )
    polar.append(r)
    polar.append(theta)
    polar.append(phi)        
    return polar

def spherToCart(r,theta,phi):
    cart = []
    x = r*math.cos(theta)*math.sin(phi)
    y = r*math.sin(theta)*math.sin(phi)
    z = r*math.cos(phi)
    cart.append(x)
    cart.append(y)
    cart.append(z)        
    return cart        

def generate_data():    
    #generate trajectory
    max_x = .2#1.5
    min_x = -.2# 0.5
    max_y = .2#1.0
    min_y = -.2#-0.9
    max_z = .2#1.5   print tr
    min_z = -.2
    margin = 0.0
    num_pt = 60 #10 # points on circle
    iterations = 1 #250 trajectories to be generated
    #open file for writing
    data_file = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_l.p","a+b")#wb
    for i in range(iterations):
        pos_x = random.random()*(max_x - margin - min_x - margin) + (min_x + margin)
        pos_y = random.random()*(max_y - margin - min_y - margin) + (min_y + margin)
        pos_z = random.random()*(max_z - margin - min_z - margin) + (min_z + margin)
        #print pos_x, pos_y, pos_z
        trial = execute_trajectory(pos_x, pos_y, pos_z, num_pt)   
        if trial is None:
            continue 
        
        print "Executed iteration:",i+1
        for data in trial:    
            pickle.dump(data, data_file)
    data_file.close()
    # get order result
    return
