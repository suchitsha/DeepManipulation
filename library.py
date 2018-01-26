# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (140, 290, -1, -1),
#  'transitions': []}
### end of header
import time
import math
import random
import cPickle as pickle

from pyutils.matrix import *
import itertools
import pandas as pd
import tensorflow as tf

def find_ik_rotational(goal, i):
    try:
        q_goal = app.rave.find_rotational_ik("right_arm", eye(4), goal, (0, 1, 0), 0, 360, 10, best=True)
    except:
        print "unable to find ik solution for offset at step %s" %i
        return None
    offset_cfg = app.rave.get_config_dict()
    offset_cfg["right_arm"] = q_goal
    return offset_cfg

def find_ik_numerically(goal, i):
    #only right arm
    q_goal = app.rave.find_ik("right_arm", goal)
    if not len(q_goal):
        print "unable to find ik solution for offset at step %s" %i
        return None
    
    offset_cfg = app.rave.get_config_dict()
    offset_cfg["right_arm"] = q_goal
    return offset_cfg

def find_ik_analytically(goal, i):
    #includes torso motion
    offset_cfg = app.rave.get_config_dict()
    try:
        q_goal = app.ikine.dual_ik(None, goal, offset_cfg)
    except:
        print "unable to find ik solution for offset at step %s" %i
        return None   
    offset_cfg.update(q_goal)
    return offset_cfg
   
def calculate_circle_positions(num_points, i, radius):
    angle = 2.0 * math.pi * (i) / num_points
        
    target_pos_x = math.cos(angle) * radius
    t_p_x = pos_x + target_pos_x
    '''if i%2:
        target_pos_y = pos_y + 0.05
    else:
        target_pos_y = pos_y - 0.05'''
    target_pos_y = math.sin(angle) * radius
    t_p_y = pos_y + target_pos_y 
    
    if i%2:
        target_pos_z = 0.03
    else:
        target_pos_z = -0.03 #'''                    
        
    t_p_z = pos_z + target_pos_z
    return t_p_z
   

def create_lissajous_positions(a, b, num_points=20, scale=1, do_plot=False):
    from numpy import sin,pi,linspace
    from pylab import plot,show,subplot

    delta = pi/2
    t = linspace(-pi, pi, num_points)
    x = sin(a * t + delta) * scale
    y = sin(b * t) * scale
    if do_plot:
        plot(x,y)
        show()
    return x, y 


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
    
   
def calc_error(g_cart, x,y,z):
    error = math.sqrt( math.pow((g_cart[0] - x),2) + math.pow((g_cart[1] - y),2) + math.pow((g_cart[2] - z),2) )
    return error  


def evaluate_random_point(clf, pos_x, pos_y, pos_z, num_points, radius):
    s_coord = cartToSpher(pos_x,pos_y,pos_z)
    sum_d = 0.0
    sum_theta = 0.0
    sum_phi = 0.0
    
    #for error
    sum_x = pos_x
    sum_y = pos_y
    sum_z = pos_z
    total_error = 0.0
    
    '''idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([0, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0.    ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)
    app.rave.remove_coords("*")
    app.rave.clear_graphic_handles("*")'''
    q_dict = app.rave.get_config_dict()    
    #tr = app.rave.get_link_frames(["rightarm1"])["rightarm1"]
    tr = app.rave.get_manip_frame("right_arm")
    #app.rave.add_coord("tr", tr)
    offset = txyz(pos_x,pos_y,pos_z)
    center_frame = dot(offset, tr)
    #TODO is this required ???
    center_frame[0:3,0:3] = dot(roty(pi/2.), rotz(pi/2))[0:3,0:3]
    app.rave.add_coord("goal", center_frame,"small")
    #print q_dict["right_arm"] 
    
    q_dict = app.find_ik_rotational(center_frame, 0)
    if q_dict is None:
        print "center frame unreachable!"
        return
    app.rave.set_config_dict(q_dict)      
    points = []
    points_expected = []
    colors = []
    motor_pos_center = q_dict["right_arm"]
    sum_motor_pos = [0.,0.,0.,0.,0.,0.,0.]
    
    l = app.create_lissajous_positions(1, 8, num_points=num_points, scale=0.1, do_plot=False)
    
    for i, (x, y) in enumerate(zip(l[0], l[1])):  
        result = clf.predict([[i, s_coord[0] , s_coord[1], s_coord[2] ]])
        
        '''
        # For cartesian positions learned
        #print 'result:' , result
        sum_d += result[0][0]
        sum_theta += result[0][1]
        sum_phi += result[0][2]
        goal_s_x = s_coord[0] + sum_d
        goal_s_y = s_coord[1] + sum_theta
        goal_s_z = s_coord[2] + sum_phi          
        goal_cart = spherToCart(goal_s_x, goal_s_y, goal_s_z)
        #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
        sum_cart = spherToCart(sum_d, sum_theta, sum_phi)
        #print 'goal:', goal_cart 
        
        
        #execute trajectory
        #do InvKin
        tr1 = center_frame
        #for others
        off = txyz(sum_cart[0], sum_cart[1], sum_cart[2])
        print off
        print tr1[0,3]
        print tr1
        goal1 = tr1 + off                
        #for circle
        #offset1 = txyz(goal_cart[0],goal_cart[1],goal_cart[2])
        #goal1 = dot(offset1, tr1)
        #app.rave.add_coord("tr1", tr1)
        app.rave.add_coord("goal1", goal1)
        #points.append(goal1[0:3,3])
        
        initial = app.rave.get_config_dict()
        #q_dict = find_ik_numerically(goal1, i)
        q_dict = app.find_ik_rotational(goal1, i)
        if q_dict is None:
            #colors.append((1, 0, 0))
            #app.rave.draw_line("line", points, colors, 4)
            return
        #qd = q_dict["right_arm"] - initial["right_arm"] 
        #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
        #    print "re-configuration not allowed"
        #    return  
          
        #colors.append((0, 1, 0))
        #app.rave.draw_line("line", points, colors, 4)
        
        app.rave.set_config_dict(q_dict)   
        #reached_frame = app.rave.get_manip_frame("right_arm")
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        #'''
        
        
        #'''
        # for motor position learned
        sum_motor_pos[0] += result[0][0]
        sum_motor_pos[1] += result[0][1]        
        sum_motor_pos[2] += result[0][2]        
        sum_motor_pos[3] += result[0][3]        
        sum_motor_pos[4] += result[0][4]        
        sum_motor_pos[5] += result[0][5]
        sum_motor_pos[6] += result[0][6]
        estimated_pos = [0.,0.,0.,0.,0.,0.,0.]
        estimated_pos[0] = motor_pos_center[0] + sum_motor_pos[0]
        estimated_pos[1] = motor_pos_center[1] + sum_motor_pos[1]
        estimated_pos[2] = motor_pos_center[2] + sum_motor_pos[2]
        estimated_pos[3] = motor_pos_center[3] + sum_motor_pos[3]
        estimated_pos[4] = motor_pos_center[4] + sum_motor_pos[4]
        estimated_pos[5] = motor_pos_center[5] + sum_motor_pos[5]
        estimated_pos[6] = motor_pos_center[6] + sum_motor_pos[6]

        #move robot to desired pose
        cdict = app.rave.get_config_dict()
        #print cdict["right_arm"]
        cdict["right_arm"] = estimated_pos
        app.rave.set_config_dict(cdict)
        
        #'''
        
        #print result
        reached_frame = app.rave.get_manip_frame("right_arm")
        points.append(reached_frame[0:3,3])
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        
                     
        #calculate error
        angle = 2.0*math.pi*(i)/num_points
        exp_x= pos_x + math.cos(angle)*radius
        '''if i%2:
            sum_y += 1/10.
        else:
            sum_y -= 1/10. '''
        exp_y = pos_y + math.sin(angle)*radius
        if i%2:
            #sum_z += 0.03
            exp_z = pos_z + 0.03   
        else:
            exp_z = pos_z - 0.03 #sum_z -= 0.03     
        #exp_z = sum_z  #+ math.sin(angle)*radius
        
        
        #current pose
        #print reached_frame
        #print reached_frame[0:3,3]
        
        expected = dot(tr, txyz(x, y, 0))
        
        error = calc_error(reached_frame[0:3,3], x, y, 0)
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        
        print "error:", error
        total_error += error
        #time.sleep(0.5)
        
        points_expected.append(expected[0:3,3])
        app.rave.draw_line("line_expected", points_expected, [0,1,0], 4)
        app.rave.draw_line("line", points, [0,0,1], 4)
    
    print "total_error:", total_error/num_points
    return

def evaluate_data_from_file(clf, data, name=""):
    num_points = len(data)
          
    #for center frame, including spherical coords, and motor pos at center
    app.idle_cfg["right_arm"] = data[0][4]
    app.rave.set_config_dict(app.idle_cfg)
    center_frame = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("goal", center_frame,"small")
    s_coord = cartToSpher(*center_frame[0:3,3])
    q_dict = app.rave.get_config_dict()  
    motor_pos_center = q_dict["right_arm"]
   
    total_error = 0.0
    points = []
    points_expected = []
    sum_motor_pos = [0.,0.,0.,0.,0.,0.,0.]
    
    #what was recorded:
    for data_point in data:
        app.idle_cfg["right_arm"] = data_point[-1]
        app.rave.set_config_dict(app.idle_cfg)
        expected = app.rave.get_manip_frame("right_arm")
        points_expected.append(expected[0:3,3])
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        app.rave.draw_line("line_expected%s" % name, points_expected, [0,1,0], 4)
        #sleep(0.5)
    
    points_expected.append(points_expected[0])
    
    #result of estimation
    for i in xrange(num_points):  
        result = clf.predict([[i, s_coord[0] , s_coord[1], s_coord[2] ]])

        # for motor position learned
        sum_motor_pos += result[0]
        estimated_pos = [0.,0.,0.,0.,0.,0.,0.]
        estimated_pos = motor_pos_center + sum_motor_pos

        #move robot to estimated pose
        cdict = app.rave.get_config_dict()
        cdict["right_arm"] = estimated_pos
        app.rave.set_config_dict(cdict)
              
        #print result
        reached_frame = app.rave.get_manip_frame("right_arm")
        points.append(reached_frame[0:3,3])
        app.rave.draw_line("line", points, [0,0,1], 4)
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  

        #error assumpotion
        error = calc_error(reached_frame[0:3,3], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        print "error:", error
        total_error += error
        
        
    print "total_error:", total_error/num_points
    return

def evaluate_data_from_file_hand(clf, data, name=""):
    num_points = len(data)
          
    #for center frame, including spherical coords, and motor pos at center
    app.idle_cfg["right_hand"] = data[0][4]
    app.rave.set_config_dict(app.idle_cfg)
    center_frame = app.rave.get_manip_frame("right_hand")
    app.rave.add_coord("goal", center_frame,"small")
    s_coord = cartToSpher(*center_frame[0:3,3])
    q_dict = app.rave.get_config_dict()  
    motor_pos_center = q_dict["right_hand"]
   
    total_error = 0.0
    points = []
    points_expected = []
    sum_motor_pos = [0.,0.,0.]
    
    #what was recorded:
    for data_point in data:
        app.idle_cfg["right_hand"] = data_point[-1]
        app.rave.set_config_dict(app.idle_cfg)
        expected = app.rave.get_manip_frame("right_hand")
        points_expected.append(expected[0:3,3])
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        app.rave.draw_line("line_expected%s" % name, points_expected, [0,1,0], 4)
        #sleep(0.5)
    
    points_expected.append(points_expected[0])
    
    #result of estimation
    for i in xrange(num_points):  
        result = clf.predict([[i, s_coord[0] , s_coord[1], s_coord[2] ]])

        # for motor position learned
        sum_motor_pos += result[0]
        estimated_pos = [0.,0.,0.]*4
        estimated_pos = motor_pos_center + sum_motor_pos

        #move robot to estimated pose
        cdict = app.rave.get_config_dict()
        cdict["right_hand"] = estimated_pos
        app.rave.set_config_dict(cdict)
              
        #print result
        reached_frame = app.rave.get_manip_frame("right_hand")
        points.append(reached_frame[0:3,3])
        app.rave.draw_line("line", points, [0,0,1], 4)
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  

        #error assumpotion
        error = calc_error(reached_frame[0:3,3], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        print "error:", error
        total_error += error
        
        
    print "total_error:", total_error/num_points
    return

def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in app.FEATURES}
    labels = tf.constant(data_set[app.LABEL].values)
    return feature_cols, labels
   
def evaluate_data_from_file_dnn(regressor_list, data, name=""):
    num_points = len(data)
          
    #for center frame, including spherical coords, and motor pos at center
    app.idle_cfg["right_arm"] = data[0][4]
    app.rave.set_config_dict(app.idle_cfg)
    center_frame = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("goal", center_frame,"small")
    s_coord = cartToSpher(*center_frame[0:3,3])
    q_dict = app.rave.get_config_dict()  
    motor_pos_center = q_dict["right_arm"]
   
    total_error = 0.0
    points = []
    points_expected = []
    sum_motor_pos = [0.,0.,0.,0.,0.,0.,0.]
    
    #what was recorded:
    for data_point in data:
        app.idle_cfg["right_arm"] = data_point[-1]
        app.rave.set_config_dict(app.idle_cfg)
        expected = app.rave.get_manip_frame("right_arm")
        points_expected.append(expected[0:3,3])
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        app.rave.draw_line("line_expected%s" % name, points_expected, [0,1,0], 4)
        #sleep(0.5)
    
    points_expected.append(points_expected[0])
    
    #result of estimation
    for i in xrange(num_points):
        pred = []
        result = []
        for j in range(len(regressor_list)): 
            list_x = [s_coord[0]]
            list_y = [s_coord[1]]
            list_z = [s_coord[2]]                        
                     
            d = {'time' : i,
             'x' : list_x ,
             'y' : list_y ,
             'z' : list_z ,
             'out': 0.0
             } 
            input_data = pd.DataFrame(d) 
            #result.append( clf.predict([[i, s_coord[0] , s_coord[1], s_coord[2] ]]) )
            y = regressor_list[j].predict(input_fn=lambda: input_fn(input_data))
            # .predict() returns an iterator; convert to a list and print predictions
            predictions = list(itertools.islice(y, 1))
            #print("Predictions: {}".format(str(predictions)))
            pred.append(predictions[0]) # because predictions[0] is a list of of size 1
        #print "pred:", pred    
        #result.append(pred)
        # for motor position learned
        sum_motor_pos[0] += pred[0]#result[0]
        sum_motor_pos[1] += pred[1]
        sum_motor_pos[2] += pred[2]
        sum_motor_pos[3] += pred[3]
        sum_motor_pos[4] += pred[4]
        sum_motor_pos[5] += pred[5]
        sum_motor_pos[6] += pred[6]
        #print "smp: " ,sum_motor_pos
        estimated_pos = [0.,0.,0.,0.,0.,0.,0.]
        estimated_pos = motor_pos_center + sum_motor_pos

        #move robot to estimated pose
        cdict = app.rave.get_config_dict()
        cdict["right_arm"] = estimated_pos
        app.rave.set_config_dict(cdict)
              
        #print result
        reached_frame = app.rave.get_manip_frame("right_arm")
        points.append(reached_frame[0:3,3])
        app.rave.draw_line("line", points, [0,0,1], 4)
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        print "reached: ",reached_frame[0:3,3] 
        print "expected: ",points_expected[i]
        #error assumpotion
        error = calc_error(reached_frame[0:3,3], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        print "error:", error
        total_error += error
        
        
    print "total_error:", total_error/num_points
    return

def evaluate_data_from_file_dnn_poly(regressor, data, name=""):
    num_points = len(data)
        
    #for center frame, including spherical coords, and motor pos at center
    app.idle_cfg["right_arm"] = data[0][4]
    app.rave.set_config_dict(app.idle_cfg)
    
    ''' To change start position manually
    a = app.rave.get_config_dict()
    for tem in range(len(a["right_arm"])):
        a["right_arm"][tem] = a["right_arm"][tem] +0.1
        print "value of a:",a["right_arm"][tem]
    app.rave.set_config_dict(a)        
    #'''
    
    center_frame = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("goal", center_frame,"small")
    s_coord = cartToSpher(*center_frame[0:3,3])
    #c_coord = [center_frame[0,3], center_frame[1,3], center_frame[2,3]]
    #TODO fix this
    #c_coord = [center_frame[0,3]-0.58, center_frame[1,3]+.29, center_frame[2,3]-0.99]
    
    ''' To change start position manually
    center_frame[0,3] = center_frame[0,3]-1.5
    center_frame[1,3] = center_frame[1,3]-0.
    center_frame[2,3] = center_frame[2,3]+0.2        
    app.rave.set_manip_frame("right_arm") = center_frame
    #'''    
    c_coord = [center_frame[0,3], center_frame[1,3], center_frame[2,3]]
    #print "c_coord:",c_coord
    
    q_dict = app.rave.get_config_dict()  
    motor_pos_center = q_dict["right_arm"]
   
    total_error = 0.0
    points = []
    points_expected = []
    sum_motor_pos = [0.,0.,0.,0.,0.,0.,0.]
    
    #what was recorded:
    for data_point in data:
        app.idle_cfg["right_arm"] = data_point[6]#data_point[-1]
        app.rave.set_config_dict(app.idle_cfg)
        expected = app.rave.get_manip_frame("right_arm")
        points_expected.append(expected[0:3,3])
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        app.rave.draw_line("line_expected%s" % name, points_expected, [0,1,0], 4)
        #sleep(0.5)
    
    points_expected.append(points_expected[0])

    sum_d = 0.0
    sum_theta = 0.0
    sum_phi = 0.0
    
    #result of estimation
    for i in xrange(num_points):
        result = []
        if app.noSpherical:
            list_x = [c_coord[0]]
            list_y = [c_coord[1]]
            list_z = [c_coord[2]]                                
        else:
            list_x = [s_coord[0]]
            list_y = [s_coord[1]]
            list_z = [s_coord[2]]                        
        if app.output == 0:
            d = {'time' : i,
                 'x' : list_x,
                 'y' : list_y,
                 'z' : list_z,
                 'out_x': 0.0, # dummy values
                 'out_y': 0.0,
                 'out_z': 0.0
                 }         
        else:             
            d = {'time' : i,
                 'x' : list_x,
                 'y' : list_y,
                 'z' : list_z,
                 'out_0': 0.0, # dummy values
                 'out_1': 0.0,
                 'out_2': 0.0,
                 'out_3': 0.0,
                 'out_4': 0.0,
                 'out_5': 0.0,
                 'out_6': 0.0                                                                              
                 } 
        input_data = pd.DataFrame(d) 
        y = regressor.predict(input_fn=lambda: input_fn(input_data))
        # .predict() returns an iterator; convert to a list and print predictions
        predictions = list(itertools.islice(y, 1))
        print("Predictions: {}".format(str(predictions)))
        if app.output == 0:
            if app.noSpherical: 
                #d,theta, phi are now actually x,y,z           
                sum_d += predictions[0][0]
                sum_theta += predictions[0][1]
                sum_phi += predictions[0][2]
                goal_s_x = c_coord[0] + sum_d
                goal_s_y = c_coord[1] + sum_theta
                goal_s_z = c_coord[2] + sum_phi          
                goal_cart = [goal_s_x, goal_s_y, goal_s_z]
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cart = [sum_d, sum_theta, sum_phi]
                
                #execute trajectory
                offset = center_frame 
                offset[0,3] = goal_cart[0]# + 0.58
                offset[1,3] = goal_cart[1]# -0.284
                offset[2,3] = goal_cart[2]# + .99
                app.rave.add_coord("offset", offset,"small")
                initial = app.rave.get_config_dict()
                q_dict = app.find_ik_rotational(offset, i)
                if q_dict is None:
                    print "cannot find IK"
                    continue#return
                #qd = q_dict["right_arm"] - initial["right_arm"] 
                #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
                #    print "re-configuration not allowed"
                #    return  
                app.rave.set_config_dict(q_dict)   
            else:
                sum_d += predictions[0][0]
                sum_theta += predictions[0][1]
                sum_phi += predictions[0][2]
                goal_s_x = s_coord[0] + sum_d
                goal_s_y = s_coord[1] + sum_theta
                goal_s_z = s_coord[2] + sum_phi          
                goal_cart = spherToCart(goal_s_x, goal_s_y, goal_s_z)
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cart = spherToCart(sum_d, sum_theta, sum_phi)
                
                #execute trajectory
                offset = center_frame 
                offset[0,3] = goal_cart[0]
                offset[1,3] = goal_cart[1]
                offset[2,3] = goal_cart[2]
                app.rave.add_coord("offset", offset,"small")
                initial = app.rave.get_config_dict()
                q_dict = app.find_ik_rotational(offset, i)
                if q_dict is None:
                    print "cannot find IK"
                    continue#return
                #qd = q_dict["right_arm"] - initial["right_arm"] 
                #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
                #    print "re-configuration not allowed"
                #    return  
                app.rave.set_config_dict(q_dict)   
        else:
            # for motor position learned
            sum_motor_pos[0] += predictions[0][0]#result[0]
            sum_motor_pos[1] += predictions[0][1]
            sum_motor_pos[2] += predictions[0][2]
            sum_motor_pos[3] += predictions[0][3]
            sum_motor_pos[4] += predictions[0][4]
            sum_motor_pos[5] += predictions[0][5]
            sum_motor_pos[6] += predictions[0][6]
            #print "smp: " ,sum_motor_pos
            estimated_pos = [0.,0.,0.,0.,0.,0.,0.]
            estimated_pos = motor_pos_center + sum_motor_pos

            #move robot to estimated pose
            cdict = app.rave.get_config_dict()
            cdict["right_arm"] = estimated_pos
            app.rave.set_config_dict(cdict)
              
        #print result
        reached_frame = app.rave.get_manip_frame("right_arm")
        points.append(reached_frame[0:3,3])
        app.rave.draw_line("line", points, [0,0,1], 4)
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        print "reached: ",reached_frame[0:3,3] 
        print "expected: ",points_expected[i]
        #error assumpotion
        error = calc_error(reached_frame[0:3,3], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        print "error:", error
        total_error += error
        
        
    print "total_error:", total_error/num_points
    return


def evaluate_data_demo_accum(regressor, n_point):
    num_points = n_point
    
    ''' To change start position manually
    a = app.rave.get_config_dict()
    for tem in range(len(a["right_arm"])):
        a["right_arm"][tem] = a["right_arm"][tem] +0.1
        print "value of a:",a["right_arm"][tem]
    app.rave.set_config_dict(a)        
    #'''
    
    center_frame = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("goal", center_frame,"small")
    s_coord = cartToSpher(*center_frame[0:3,3])
    #c_coord = [center_frame[0,3]-0.58, center_frame[1,3]+.29, center_frame[2,3]-0.99]
    
    ''' To change start position manually
    center_frame[0,3] = center_frame[0,3]-1.5
    center_frame[1,3] = center_frame[1,3]-0.
    center_frame[2,3] = center_frame[2,3]+0.2        
    app.rave.set_manip_frame("right_arm") = center_frame
    #'''    
    c_coord = [center_frame[0,3], center_frame[1,3], center_frame[2,3]]
    #print "c_coord:",c_coord
    
    q_dict = app.rave.get_config_dict()  
    motor_pos_center = q_dict["right_arm"]
   
    total_error = 0.0
    points = []
    points_expected = []
    sum_motor_pos = [0.,0.,0.,0.,0.,0.,0.]
    
    sum_d = 0.0
    sum_theta = 0.0
    sum_phi = 0.0
    
    #result of estimation
    for i in xrange(num_points):
        result = []
        if app.noSpherical:
            list_x = [c_coord[0]]
            list_y = [c_coord[1]]
            list_z = [c_coord[2]]                                
        else:
            list_x = [s_coord[0]]
            list_y = [s_coord[1]]
            list_z = [s_coord[2]]                        
        if app.output == 0:
            d = {'time' : i,
                 'x' : list_x,
                 'y' : list_y,
                 'z' : list_z,
                 'out_x': 0.0, # dummy values
                 'out_y': 0.0,
                 'out_z': 0.0
                 }         
        else:             
            d = {'time' : i,
                 'x' : list_x,
                 'y' : list_y,
                 'z' : list_z,
                 'out_0': 0.0, # dummy values
                 'out_1': 0.0,
                 'out_2': 0.0,
                 'out_3': 0.0,
                 'out_4': 0.0,
                 'out_5': 0.0,
                 'out_6': 0.0                                                                              
                 } 
        input_data = pd.DataFrame(d) 
        y = regressor.predict(input_fn=lambda: input_fn(input_data))
        # .predict() returns an iterator; convert to a list and print predictions
        predictions = list(itertools.islice(y, 1))
        print("Predictions: {}".format(str(predictions)))
        if app.output == 0:
            if app.noSpherical: 
                #d,theta, phi are now actually x,y,z           
                sum_d += predictions[0][0]
                sum_theta += predictions[0][1]
                sum_phi += predictions[0][2]
                goal_s_x = c_coord[0] + sum_d
                goal_s_y = c_coord[1] + sum_theta
                goal_s_z = c_coord[2] + sum_phi          
                goal_cart = [goal_s_x, goal_s_y, goal_s_z]
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cart = [sum_d, sum_theta, sum_phi]
                
                #execute trajectory
                offset = center_frame 
                offset[0,3] = goal_cart[0]# + 0.58
                offset[1,3] = goal_cart[1]# -0.284
                offset[2,3] = goal_cart[2]# + .99
                app.rave.add_coord("offset", offset,"small")
                initial = app.rave.get_config_dict()
                q_dict = app.find_ik_rotational(offset, i)
                if q_dict is None:
                    print "cannot find IK"
                    continue#return
                #qd = q_dict["right_arm"] - initial["right_arm"] 
                #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
                #    print "re-configuration not allowed"
                #    return  
                app.rave.set_config_dict(q_dict)   
            else:
                sum_d += predictions[0][0]
                sum_theta += predictions[0][1]
                sum_phi += predictions[0][2]
                goal_s_x = s_coord[0] + sum_d
                goal_s_y = s_coord[1] + sum_theta
                goal_s_z = s_coord[2] + sum_phi          
                goal_cart = spherToCart(goal_s_x, goal_s_y, goal_s_z)
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cart = spherToCart(sum_d, sum_theta, sum_phi)
                
                #execute trajectory
                offset = center_frame 
                offset[0,3] = goal_cart[0]
                offset[1,3] = goal_cart[1]
                offset[2,3] = goal_cart[2]
                app.rave.add_coord("offset", offset,"small")
                initial = app.rave.get_config_dict()
                q_dict = app.find_ik_rotational(offset, i)
                if q_dict is None:
                    print "cannot find IK"
                    continue#return
                #qd = q_dict["right_arm"] - initial["right_arm"] 
                #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
                #    print "re-configuration not allowed"
                #    return  
                app.rave.set_config_dict(q_dict)   
        else:
            # for motor position learned
            sum_motor_pos[0] += predictions[0][0]#result[0]
            sum_motor_pos[1] += predictions[0][1]
            sum_motor_pos[2] += predictions[0][2]
            sum_motor_pos[3] += predictions[0][3]
            sum_motor_pos[4] += predictions[0][4]
            sum_motor_pos[5] += predictions[0][5]
            sum_motor_pos[6] += predictions[0][6]
            #print "smp: " ,sum_motor_pos
            estimated_pos = [0.,0.,0.,0.,0.,0.,0.]
            estimated_pos = motor_pos_center + sum_motor_pos

            #move robot to estimated pose
            cdict = app.rave.get_config_dict()
            cdict["right_arm"] = estimated_pos
            app.rave.set_config_dict(cdict)
              
        #print result
        reached_frame = app.rave.get_manip_frame("right_arm")
        points.append(reached_frame[0:3,3])
        app.rave.draw_line("line", points, [0,0,1], 4)
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        print "reached: ",reached_frame[0:3,3] 
        #print "expected: ",points_expected[i]
        #error assumption
        #error = calc_error(reached_frame[0:3,3], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        #print "error:", error
        #total_error += error
        
        
    #print "total_error:", total_error/num_points
    return

def evaluate_data_demo_broom(regressorL, regressorR, n_point):
    num_points = n_point
    
    ''' To change start position manually
    a = app.rave.get_config_dict()
    for tem in range(len(a["right_arm"])):
        a["right_arm"][tem] = a["right_arm"][tem] +0.1
        print "value of a:",a["right_arm"][tem]
    app.rave.set_config_dict(a)        
    #'''
    
    center_frameR = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("goalR", center_frameR,"small")
    s_coordR = cartToSpher(*center_frameR[0:3,3])
    #c_coord = [center_frame[0,3]-0.58, center_frame[1,3]+.29, center_frame[2,3]-0.99]
    
    
    center_frameL = app.rave.get_manip_frame("left_arm")
    app.rave.add_coord("goalL", center_frameL,"small")
    s_coordL = cartToSpher(*center_frameL[0:3,3])
    
    ''' To change start position manually
    center_frame[0,3] = center_frame[0,3]-1.5
    center_frame[1,3] = center_frame[1,3]-0.
    center_frame[2,3] = center_frame[2,3]+0.2        
    app.rave.set_manip_frame("right_arm") = center_frame
    #'''    
    c_coordR = [center_frameR[0,3], center_frameR[1,3], center_frameR[2,3]]
    c_coordL = [center_frameL[0,3], center_frameL[1,3], center_frameL[2,3]]
    
    #print "c_coord:",c_coord
    
    q_dict = app.rave.get_config_dict()  
    motor_pos_centerR = q_dict["right_arm"]
    motor_pos_centerL = q_dict["left_arm"]


    total_errorR = 0.0
    pointsR = []
    points_expectedR = []
    sum_motor_posR = [0.,0.,0.,0.,0.,0.,0.]
    
    sum_dR = 0.0
    sum_thetaR = 0.0
    sum_phiR = 0.0
    
    total_errorL = 0.0
    pointsL = []
    points_expectedL = []
    sum_motor_posL = [0.,0.,0.,0.,0.,0.,0.]
    
    sum_dL = 0.0
    sum_thetaL = 0.0
    sum_phiL = 0.0

    #result of estimation
    for i in xrange(num_points):
        resultR = []
        resultL = []        
        if app.noSpherical:
            list_xR = [c_coordR[0]]
            list_yR = [c_coordR[1]]
            list_zR = [c_coordR[2]]                                
            list_xL = [c_coordL[0]]
            list_yL = [c_coordL[1]]
            list_zL = [c_coordL[2]]                                
            
        else:
            list_xR = [s_coordR[0]]
            list_yR = [s_coordR[1]]
            list_zR = [s_coordR[2]]                        
            list_xL = [s_coordL[0]]
            list_yL = [s_coordL[1]]
            list_zL = [s_coordL[2]]                        
            
        if app.output == 0:
            dR = {'time' : i + 40,
                 'x' : list_xR,
                 'y' : list_yR,
                 'z' : list_zR,
                 'out_x': 0.0, # dummy values
                 'out_y': 0.0,
                 'out_z': 0.0
                 }         
            dL = {'time' : i + 40,
                 'x' : list_xL,
                 'y' : list_yL,
                 'z' : list_zL,
                 'out_x': 0.0, # dummy values
                 'out_y': 0.0,
                 'out_z': 0.0
                 }         


        else:             
            dR = {'time' : i + 40,
                 'x' : list_xR,
                 'y' : list_yR,
                 'z' : list_zR,
                 'out_0': 0.0, # dummy values
                 'out_1': 0.0,
                 'out_2': 0.0,
                 'out_3': 0.0,
                 'out_4': 0.0,
                 'out_5': 0.0,
                 'out_6': 0.0                                                                              
                 } 
                 
            dL = {'time' : i + 40,
                 'x' : list_xL,
                 'y' : list_yL,
                 'z' : list_zL,
                 'out_0': 0.0, # dummy values
                 'out_1': 0.0,
                 'out_2': 0.0,
                 'out_3': 0.0,
                 'out_4': 0.0,
                 'out_5': 0.0,
                 'out_6': 0.0                                                                              
                 }                  
        input_dataR = pd.DataFrame(dR) 
        input_dataL = pd.DataFrame(dL) 
        yR = regressorR.predict(input_fn=lambda: input_fn(input_dataR))
        yL = regressorL.predict(input_fn=lambda: input_fn(input_dataL))
        # .predict() returns an iterator; convert to a list and print predictions
        predictionsR = list(itertools.islice(yR, 1))
        print("Predictions: {}".format(str(predictionsR)))
        predictionsL = list(itertools.islice(yL, 1))
        print("Predictions: {}".format(str(predictionsL)))
        if app.output == 0:
            if app.noSpherical: 
                #d,theta, phi are now actually x,y,z           
                sum_dR += predictionsR[0][0]
                sum_thetaR += predictionsR[0][1]
                sum_phiR += predictionsR[0][2]
                goal_s_xR = c_coordR[0] + sum_dR
                goal_s_yR = c_coordR[1] + sum_thetaR
                goal_s_zR = c_coordR[2] + sum_phiR          
                goal_cartR = [goal_s_xR, goal_s_yR, goal_s_zR]
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cartR = [sum_dR, sum_thetaR, sum_phiR]
                
                sum_dL += predictionsL[0][0]
                sum_thetaL += predictionsL[0][1]
                sum_phiL += predictionsL[0][2]
                goal_s_xL = c_coordL[0] + sum_dL
                goal_s_yL = c_coordL[1] + sum_thetaL
                goal_s_zL = c_coordL[2] + sum_phiL          
                goal_cartL = [goal_s_xL, goal_s_yL, goal_s_zL]
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cartL = [sum_dL, sum_thetaL, sum_phiL]



                #execute trajectory
                offsetR = center_frameR 
                offsetR[0,3] = goal_cartR[0]# + 0.58
                offsetR[1,3] = goal_cartR[1]# -0.284
                offsetR[2,3] = goal_cartR[2]# + .99
                
                offsetL = center_frameL 
                offsetL[0,3] = goal_cartL[0]# + 0.58
                offsetL[1,3] = goal_cartL[1]# -0.284
                offsetL[2,3] = goal_cartL[2]# + .99

                app.rave.add_coord("offsetR", offsetR,"small")                
                app.rave.add_coord("offsetL", offsetL,"small")

                initial = app.rave.get_config_dict()
                q_dictR = app.find_ik_rotational(offsetR, i)
                if q_dictR is None:
                    print "cannot find IK"
                    continue#return

                q_dictL = app.find_ik_rotational(offsetL, i)
                if q_dictL is None:
                    print "cannot find IK"
                    continue#return


                #qd = q_dict["right_arm"] - initial["right_arm"] 
                #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
                #    print "re-configuration not allowed"
                #    return  
                #TODO fix this it should be combined value of qdictR and qdictL , solution is in the 5-10 lines above from where this library method is called
                app.rave.set_config_dict(q_dict)   
            else:
                #TODO not fixed this section for two arms
                sum_d += predictions[0][0]
                sum_theta += predictions[0][1]
                sum_phi += predictions[0][2]
                goal_s_x = s_coord[0] + sum_d
                goal_s_y = s_coord[1] + sum_theta
                goal_s_z = s_coord[2] + sum_phi          
                goal_cart = spherToCart(goal_s_x, goal_s_y, goal_s_z)
                #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
                sum_cart = spherToCart(sum_d, sum_theta, sum_phi)
                
                #execute trajectory
                offset = center_frame 
                offset[0,3] = goal_cart[0]
                offset[1,3] = goal_cart[1]
                offset[2,3] = goal_cart[2]
                app.rave.add_coord("offset", offset,"small")
                initial = app.rave.get_config_dict()
                q_dict = app.find_ik_rotational(offset, i)
                if q_dict is None:
                    print "cannot find IK"
                    continue#return
                #qd = q_dict["right_arm"] - initial["right_arm"] 
                #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
                #    print "re-configuration not allowed"
                #    return  
                app.rave.set_config_dict(q_dict)   
        else:
            # for motor position learned
            sum_motor_posR[0] += predictionsR[0][0]#result[0]
            sum_motor_posR[1] += predictionsR[0][1]
            sum_motor_posR[2] += predictionsR[0][2]
            sum_motor_posR[3] += predictionsR[0][3]
            sum_motor_posR[4] += predictionsR[0][4]
            sum_motor_posR[5] += predictionsR[0][5]
            sum_motor_posR[6] += predictionsR[0][6]
            #print "smp: " ,sum_motor_pos
            estimated_posR = [0.,0.,0.,0.,0.,0.,0.]
            estimated_posR = motor_pos_centerR + sum_motor_posR

            sum_motor_posL[0] += predictionsL[0][0]#result[0]
            sum_motor_posL[1] += predictionsL[0][1]
            sum_motor_posL[2] += predictionsL[0][2]
            sum_motor_posL[3] += predictionsL[0][3]
            sum_motor_posL[4] += predictionsL[0][4]
            sum_motor_posL[5] += predictionsL[0][5]
            sum_motor_posL[6] += predictionsL[0][6]
            #print "smp: " ,sum_motor_pos
            estimated_posL = [0.,0.,0.,0.,0.,0.,0.]
            estimated_posL = motor_pos_centerL + sum_motor_posL


            #move robot to estimated pose
            cdict = app.rave.get_config_dict()
            cdict["right_arm"] = estimated_posR
            cdict["left_arm"] = estimated_posL
            app.rave.set_config_dict(cdict)
              
        #print result
        reached_frameR = app.rave.get_manip_frame("right_arm")
        reached_frameL = app.rave.get_manip_frame("left_arm")        
        pointsR.append(reached_frameR[0:3,3])
        app.rave.draw_line("lineR", pointsR, [0,0,1], 4)
        pointsL.append(reached_frameL[0:3,3])
        app.rave.draw_line("lineL", pointsL, [0,0,1], 4)
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
        print "reachedR: ",reached_frameR[0:3,3] 
        print "reachedL: ",reached_frameL[0:3,3] 
        #print "expected: ",points_expected[i]
        #error assumption
        #error = calc_error(reached_frame[0:3,3], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        #print "error:", error
        #total_error += error
        
        
    #print "total_error:", total_error/num_points
    return


def init(self):
    app.find_ik_analytically = find_ik_analytically
    app.find_ik_numerically = find_ik_numerically
    app.find_ik_rotational = find_ik_rotational
    app.create_lissajous_positions = create_lissajous_positions
    app.evaluate_data_from_file = evaluate_data_from_file
    app.evaluate_random_point = evaluate_random_point
    app.evaluate_data_from_file_dnn = evaluate_data_from_file_dnn
    app.evaluate_data_from_file_dnn_poly = evaluate_data_from_file_dnn_poly
    app.evaluate_data_from_file_hand = evaluate_data_from_file_hand
    app.evaluate_data_demo_broom = evaluate_data_demo_broom
    app.spherToCart = spherToCart
    app.cartToSpher = cartToSpher
    app.evaluate_data_demo_accum = evaluate_data_demo_accum
def execute(self):
    print "execute of %s called!" % self.name
    
    #tests:
    '''
    app.rave.clear_graphic_handles("*")
    app.rave.remove_coords("*")
    l = app.create_lissajous_positions(1, 2, num_points=20, scale=0.1, do_plot=False)
    for i, (x, y) in enumerate(zip(l[0], l[1])):
        mf = app.rave.get_manip_frame("right_arm")
        #app.rave.add_coord("lissajous%d" % i, dot(txyz(*mf[0:3,3]), txyz(x, y, 0)))
        app.rave.add_coord("lissajous%d" % i, dot(mf, txyz(x, y, 0)))
    '''
