# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 370, -1, -1),
#  'transitions': []}
### end of header
import os
import sys
import pprint
import traceback
import numpy as np
import json
import cPickle as pickle
from os import listdir
from os.path import isfile, join
from pyutils.matrix import *
import matplotlib.pyplot as plt

lnrecorder_python_path = "/volume/USERSTORE/leid_da/externals/links_and_nodes/examples/lnrecorder/python"
#lnrecorder_python_path = "/home/that/fs/ln_base/examples/lnrecorder/python"
if lnrecorder_python_path not in sys.path:
    sys.path.insert(0, lnrecorder_python_path)
    
from lnrecorder import *
   
def execute(self):
    base_dr = "/home_local/shar_sc/data_from_robot/new_straight/180"
    #base_dr = "/home_local/shar_sc/data_from_robot/scrubbing1"
    #base_dr = "/home/that/fs/data/experiment_logs/straight_180/"
    #read directory
    lst_of_files = [join(base_dr, fn) for fn in listdir(base_dr) if isfile(join(base_dr, fn))]
    #process image    
    for i in lst_of_files:
        app.log_file = i
        print "current file is: %s" %app.log_file
        replay_and_save()
    print "processed all the files in the directory"    
    #app.log_file = "/home_local/shar_sc/data_from_robot/random/random_20170808_170701.log"
    #"/home/that/fs/data/experiment_logs/circle_20170630_100532" # "/home/shar_sc/Documents/DirtDetection/data/motion/demonstration/circle1"
    # "/volume/USERSTORE/that/data/suchit/suchit_20170518_172404"
    return    
        
def replay_and_save():
    print "loading", app.log_file
    #TODO close file
    #TODO for now we divide the trajectory by number of packets
    try:
        log = lnrecorder_log(app.log_file) 
    except:
        return
    #app.json_dump = []
    motor_pos = []
    torso = []
    force = []#
    use_the_force = False
    table_height = False
    try:
        counter = 0                        
        for i, pkt in enumerate(log.iter_packets()):
            sleep(0.000001)                        
            packet = log.decode_packet(pkt, with_meta=False)
            #print "packet", packet
            if packet["name"] == "alfred.telemetry":
                counter += 1
                if counter % 10: #%100 for first round of recordings
                    continue
                #print i    
                #print packet["header"]
                data = packet["data"]
                f = data["f_spring_right"][1]
                if use_the_force:
                    if f >= -8.5: 
                        continue
                force.append( f )
                
                q_cmd = data["q_cmd"]
                cfg = app.rave.get_config_dict()
                cfg["torso"] = q_cmd[0:3]
                cfg["right_arm"] = q_cmd[3:10]
                cfg["left_arm"] = q_cmd[10:17]
                cfg["head"] = q_cmd[17:19]
                app.rave.set_config_dict(cfg)
                #sleep(0.5)
                #app.rave.set_robot_config(q_cmd, range(19))
                
                r_frame = app.rave.get_manip_frame("right_arm")
                t_height = r_frame[0:3,3]
                #print t_height
                if table_height:
                    if t_height[2] > 0.89:  #0:905, 45:89, 90:905, 135:90, 180: 89
                        continue
                    
                motor_pos.append(cfg["right_arm"])
                torso.append(cfg["torso"])
            #else:
            #    continue
        print "packet count:",counter
        if use_the_force:
            if not force:
                return
        #for removing packets from the begining and the end
        '''
        l_mp = len(motor_pos)
        l_ts = len(torso)
        motor_pos = motor_pos[int(l_mp/3):int(2*l_mp/3)]
        torso = torso[int(l_ts/4):int(2*l_ts/4)]
        #'''    
        trial = save_data(motor_pos,torso)
        data_file = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_cnn1.p","a+b")#data_rec_accum2.p","a+b")#wb
        for dat in trial:    
            pickle.dump(dat, data_file)
        data_file.close()
        print "finished writing data"
    except:
        print traceback.format_exc()
    return

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
    
    
def save_data(motor_pos,torso):
     
    cdict = app.rave.get_config_dict()
    cdict["right_arm"] = motor_pos[0]
    cdict["torso"] = torso[0]
    app.rave.set_config_dict(cdict)
    current_frame = app.rave.get_manip_frame("right_arm")
    
    trial = []    
    pos = cartToSpher(current_frame[0,3],current_frame[1,3],current_frame[2,3])
    pos_cart = [current_frame[0,3],current_frame[1,3],current_frame[2,3]]
    motor_pos_center = motor_pos[0]
    torso_center = torso[0]
    
    goal_pos_prev = pos
    goal_pos_prev_cart = pos_cart
    motor_pos_prev = motor_pos_center
    torso_prev = torso_center
    
    name = "path"
    points_expected = []    
    app.rave.clear_graphic_handles("*")

    step = len(motor_pos)/10. #TODO this should be 10 
    for i in range(10):
        index = int(step*(i)+1)#(i+1))
        cdict["right_arm"] = motor_pos[index-1]
        cdict["torso"] = torso[index-1]
        
        app.rave.set_config_dict(cdict)
        current_frame = app.rave.get_manip_frame("right_arm")
        
        #draw
        points_expected.append(current_frame[0:3,3])
        #print points_expected
        app.rave.draw_line("line_expected%s" % name, points_expected, [0,1,0], 4)
                    
        goal_pos = cartToSpher(current_frame[0,3],current_frame[1,3],current_frame[2,3])
        goal_delta = [goal_pos[0] - goal_pos_prev[0], goal_pos[1] - goal_pos_prev[1], goal_pos[2] - goal_pos_prev[2]]
        goal_pos_prev = goal_pos
        
        goal_pos_cart = [current_frame[0,3],current_frame[1,3],current_frame[2,3]]
        goal_delta_cart = [goal_pos_cart[0] - goal_pos_prev_cart[0], goal_pos_cart[1] - goal_pos_prev_cart[1], goal_pos_cart[2] - goal_pos_prev_cart[2]]
        goal_pos_prev_cart = goal_pos_cart
        
        motor_pos_step = motor_pos[index-1]
        motor_pos_delta = [motor_pos_step[0] - motor_pos_prev[0], motor_pos_step[1] - motor_pos_prev[1], motor_pos_step[2] - motor_pos_prev[2], motor_pos_step[3] - motor_pos_prev[3], motor_pos_step[4] - motor_pos_prev[4], motor_pos_step[5] - motor_pos_prev[5], motor_pos_step[6] - motor_pos_prev[6] ]
        motor_pos_prev = motor_pos_step
        
        torso_step = torso[index-1]
        torso_motor_delta = [torso_step[0] - torso_prev[0], torso_step[1] - torso_prev[1], torso_step[2] - torso_prev[2] ]
        torso_prev = torso_step
        tim = i + 80 
        trial.append([tim, pos, goal_delta, goal_pos, motor_pos_center, motor_pos_delta, motor_pos_step, pos_cart, goal_delta_cart, goal_pos_cart, torso_center,torso_motor_delta,torso_step ])
    #sleep(5)
    #print trial    
    return trial
