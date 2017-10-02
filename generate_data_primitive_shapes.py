# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 530, -1, -1),
#  'transitions': []}
### end of header
import time
import math
import random
import cPickle as pickle

from pyutils.matrix import *

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    generate_data()

def execute_trajectory(pos_x, pos_y, pos_z, radius, num_points):

    idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([0., -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0., 0.4, 0.2, 0., 0.4, 0.2, 0., 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0., 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0. ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)

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
    
    target_pos_x = 0.0
    target_pos_y = 0.0
    target_pos_z = 0.0  
          
    for i in range( 22 + 16 + 12 ): # for circle, square, triangle.

        if (i < 11): # circle
            #print "circle:", i
            angle = 2.0 * math.pi * (i) / num_points
            target_pos_x = math.cos(angle) * radius
            t_p_x = pos_x + target_pos_x
            target_pos_y = math.sin(angle) * radius
            t_p_y = pos_y + target_pos_y             
            target_pos_z = 0.0                
            t_p_z = pos_z + target_pos_z

        elif (i >= 11 and i < 22): # circle in anti-clockwise direction
            #print "circle2:", i
            angle = 2.0 * math.pi * (i) / num_points
            target_pos_x = -math.cos(angle) * radius
            t_p_x = pos_x + target_pos_x
            target_pos_y = math.sin(angle) * radius
            t_p_y = pos_y + target_pos_y 
            target_pos_z = 0.0                   
            t_p_z = pos_z + target_pos_z        
        elif ( i >=22   and i < 30 ): #square max(abs(x),abs(y)) = c
            #print "square:", i
            c = square(i, radius)
            target_pos_x += c[0]
            target_pos_y += c[1]
            target_pos_z += c[2]                        
            t_p_x = pos_x + target_pos_x            
            t_p_y = pos_y + target_pos_y 
            t_p_z = pos_z + target_pos_z            
        elif ( i >=30   and i < 38 ): #square anti-clockwise max(abs(x),abs(y)) = c
            #print "square2:", i
            c = square2(i, radius)
            target_pos_x += c[0]
            target_pos_y += c[1]
            target_pos_z += c[2]                        
            t_p_x = pos_x + target_pos_x            
            t_p_y = pos_y + target_pos_y 
            t_p_z = pos_z + target_pos_z            

        elif ( i >=38   and i < 44 ): #triangle
            #print "triangle:", i                            
            c = triangle(i, radius)
            target_pos_x += c[0]
            target_pos_y += c[1]
            target_pos_z += c[2]                        
            t_p_x = pos_x + target_pos_x            
            t_p_y = pos_y + target_pos_y 
            t_p_z = pos_z + target_pos_z            
            
        elif ( i >=44   and i < 50 ): #triangle anti-clockwise
            #print "triangle2:", i                            
            c = triangle2(i, radius)
            target_pos_x += c[0]
            target_pos_y += c[1]
            target_pos_z += c[2]                        
            t_p_x = pos_x + target_pos_x            
            t_p_y = pos_y + target_pos_y 
            t_p_z = pos_z + target_pos_z            
            
        #execute trajectory
        #do InvKin
        tr1 = center_frame
        offset1 = txyz(target_pos_x,target_pos_y,target_pos_z)
        #offset1 = txyz(0.,0.,0.)
        goal1 = dot(offset1, tr1)
        points.append(goal1[0:3,3])
        
        initial = app.rave.get_config_dict()
        #q_dict = app.find_ik_numerically(goal1, i)
        q_dict = app.find_ik_rotational(goal1, i)
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
        app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        #sleep(0.2) 

        #prepare data
        time= i
        pos = cartToSpher(pos_x,pos_y,pos_z)
        goal_pos = cartToSpher(t_p_x,t_p_y,t_p_z)
        goal_delta = [goal_pos[0] - goal_pos_prev[0], goal_pos[1] - goal_pos_prev[1], goal_pos[2] - goal_pos_prev[2]]
        goal_pos_prev = goal_pos
        
        #cartesian version
        pos_cart = [pos_x,pos_y,pos_z]
        goal_pos_cart = [t_p_x,t_p_y,t_p_z]
        goal_delta_cart = [goal_pos_cart[0] - goal_pos_prev_cart[0], goal_pos_cart[1] - goal_pos_prev_cart[1], goal_pos_cart[2] - goal_pos_prev_cart[2]]
        goal_pos_prev_cart = goal_pos_cart
        
        motor_pos = q_dict["right_arm"]
        motor_pos_delta = [motor_pos[0] - motor_pos_prev[0], motor_pos[1] - motor_pos_prev[1], motor_pos[2] - motor_pos_prev[2], motor_pos[3] - motor_pos_prev[3], motor_pos[4] - motor_pos_prev[4], motor_pos[5] - motor_pos_prev[5], motor_pos[6] - motor_pos_prev[6] ]
        motor_pos_prev = motor_pos
        
        trial.append([time, pos, goal_delta, goal_pos, motor_pos_center, motor_pos_delta, motor_pos, pos_cart, goal_delta_cart, goal_pos_cart ])
        #print [time, pos, goal_delta, goal_pos, motor_pos_center, motor_pos_delta, motor_pos, pos_cart, goal_delta_cart, goal_pos_cart ]
        #print "data:" , data        
        #print data
        #time.sleep(1)
        #write data


    return trial

# super hard coded 
#square of side 2c clockwise
def square(i,c):
    if (i == 22):
        x = 2*c/2.
        y = 0.
        z = 0.
    if (i == 23):
        x = 2*c/4.
        y = -2*c/4.
        z = 0.
    if (i == 24):
        x = 0.
        y = -2*c/2.
        z = 0.
    if (i == 25):
        x = -2*c/4.
        y = -2*c/4.
        z = 0.
    if (i == 26):
        x = -2*c/2.
        y = 0.
        z = 0.        
    if (i == 27):
        x = -2*c/4.
        y = 2*c/4.
        z = 0. 
    if (i == 28):
        x = 0.
        y = 2*c/2.
        z = 0.        
    if (i == 29):
        x = 2*c/4.
        y = 2*c/4.
        z = 0. 
    return [x, y, z ]

# super hard coded 
#square of side 2c anticlockwise
def square2(i,c):
    if (i == 30):
        x = -2*c/2.
        y = 0.
        z = 0.
    if (i == 31):
        x = -2*c/4.
        y = -2*c/4.
        z = 0.
    if (i == 32):
        x = 0.
        y = -2*c/2.
        z = 0.
    if (i == 33):
        x = 2*c/4.
        y = -2*c/4.
        z = 0.
    if (i == 34):
        x = 2*c/2.
        y = 0.
        z = 0.        
    if (i == 35):
        x = 2*c/4.
        y = 2*c/4.
        z = 0. 
    if (i == 36):
        x = 0.
        y = 2*c/2.
        z = 0.        
    if (i == 37):
        x = -2*c/4.
        y = 2*c/4.
        z = 0. 
    return [x, y, z ]
        
# super hard coded 
#equilatral triangle of side 2c clockwise
def triangle(i,c):
    if (i == 38):
        x = 2*c/4.
        y = 0.
        z = 0.
    if (i == 39):
        x = (2*c/2.)*math.cos(60.*math.pi/180.)
        y = (-2*c/2.)*math.sin(60.*math.pi/180.)
        z = 0.
    if (i == 40):
        x = (-2*c/4.)*math.sin(30.*math.pi/180.)
        y = (-2*c/4.)*math.cos(30.*math.pi/180.)
        z = 0.
    if (i == 41):
        x = -2*c/2.
        y = 0.
        z = 0.
    if (i == 42):
        x = -2*c/8.
        y = (2*c/4.)*math.sin(30.*math.pi/180.)
        z = 0.        
    if (i == 43):
        x = (2*c/2.)*math.cos(60.*math.pi/180.)
        y = (2*c/2.)*math.sin(60.*math.pi/180.)
        z = 0.               
    return [x, y, z ]

# super hard coded 
#equilatral triangle of side 2c anti-clockwise
def triangle2(i,c):
    if (i == 44):
        x = -2*c/4.
        y = 0.
        z = 0.
    if (i == 45):
        x = (-2*c/2.)*math.cos(60.*math.pi/180.)
        y = (-2*c/2.)*math.sin(60.*math.pi/180.)
        z = 0.
    if (i == 46):
        x = (2*c/4.)*math.sin(30.*math.pi/180.)
        y = (-2*c/4.)*math.cos(30.*math.pi/180.)
        z = 0.
    if (i == 47):
        x = 2*c/2.
        y = 0.
        z = 0.
    if (i == 48):
        x = 2*c/8.
        y = (2*c/4.)*math.sin(30.*math.pi/180.)
        z = 0.        
    if (i == 49):
        x = (-2*c/2.)*math.cos(60.*math.pi/180.)
        y = (2*c/2.)*math.sin(60.*math.pi/180.)
        z = 0.               
    return [x, y, z ]
    
    
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
    radius = 0.1
    num_points = 10 #10 # points on circle
    iterations = 200  #100-120 per hour #1500 # trajectories to be generated
    #open file for writing
    data_file = open("/home/shar_sc/Documents/DirtDetection/data/motion/data.p","a+b")#wb
    for i in range(iterations):
        pos_x = random.random()*(max_x - margin - min_x - margin) + (min_x + margin)
        pos_y = random.random()*(max_y - margin - min_y - margin) + (min_y + margin)
        pos_z = random.random()*(max_z - margin - min_z - margin) + (min_z + margin)
        #print pos_x, pos_y, pos_z
        trial = execute_trajectory(pos_x, pos_y, pos_z, radius, num_points)   
        if trial is None:
            continue 
        
        print "Executed iteration:",i+1
        for data in trial:    
            pickle.dump(data, data_file)
    data_file.close()
    return
