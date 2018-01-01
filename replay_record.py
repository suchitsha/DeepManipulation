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

from pyutils.matrix import *

lnrecorder_python_path = "/volume/USERSTORE/leid_da/externals/links_and_nodes/examples/lnrecorder/python"
#lnrecorder_python_path = "/home/that/fs/ln_base/examples/lnrecorder/python"
if lnrecorder_python_path not in sys.path:
    sys.path.insert(0, lnrecorder_python_path)
    
from lnrecorder import *

app.log_file = "/home/that/fs/data/experiment_logs/circle_20170630_100532" # "/home/shar_sc/Documents/DirtDetection/data/motion/demonstration/circle1"
# "/volume/USERSTORE/that/data/suchit/suchit_20170518_172404"
print app.log_file    
def execute(self):
    print "loading", app.log_file
    
    #TODO close file
    #TODO for now we divide the trajectory by number of packets
    log = lnrecorder_log(app.log_file) 
    idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([0, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0. ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)

    #app.json_dump = []
    motor_pos = []
    try:
        counter = 0
        for i, pkt in enumerate(log.iter_packets()):
            sleep(0.000001)            
            packet = log.decode_packet(pkt, with_meta=False)
            if packet["name"] == "alfred.telemetry":
                counter += 1
                if counter % 100:
                    continue
                #print packet["header"]
                data = packet["data"]
                q_cmd = data["q_cmd"]
                cfg = app.rave.get_config_dict()
                cfg["torso"] = q_cmd[0:3]
                cfg["right_arm"] = q_cmd[3:10]
                cfg["left_arm"] = q_cmd[10:17]
                cfg["head"] = q_cmd[17:19]
                app.rave.set_config_dict(cfg)
                #app.rave.set_robot_config(q_cmd, range(19))
                motor_pos.append(cfg["right_arm"])
            #else:
            #    continue
        #print i
        trial = save_data(motor_pos)
        data_file = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_circle.p","a+b")#wb
        for dat in trial:    
            pickle.dump(dat, data_file)
        data_file.close()
        print "finished writing data"
    except:
        print traceback.format_exc()
    return


def save_data(motor_pos):
     
    cdict = app.rave.get_config_dict()
    cdict["right_arm"] = motor_pos[0]
    app.rave.set_config_dict(cdict)
    current_frame = app.rave.get_manip_frame("right_arm")
    
    trial = []    
    pos = app.cartToSpher(current_frame[0,3],current_frame[1,3],current_frame[2,3])
    pos_cart = [current_frame[0,3],current_frame[1,3],current_frame[2,3]]
    motor_pos_center = motor_pos[0]
    
    goal_pos_prev = pos
    goal_pos_prev_cart = pos_cart
    motor_pos_prev = motor_pos_center
    
    step = len(motor_pos)/10.
    for i in range(10):
        index = int(step*(i+1))
        cdict["right_arm"] = motor_pos[index-1]
        app.rave.set_config_dict(cdict)
        current_frame = app.rave.get_manip_frame("right_arm")
        
        goal_pos = app.cartToSpher(current_frame[0,3],current_frame[1,3],current_frame[2,3])
        goal_delta = [goal_pos[0] - goal_pos_prev[0], goal_pos[1] - goal_pos_prev[1], goal_pos[2] - goal_pos_prev[2]]
        goal_pos_prev = goal_pos
        
        goal_pos_cart = [current_frame[0,3],current_frame[1,3],current_frame[2,3]]
        goal_delta_cart = [goal_pos_cart[0] - goal_pos_prev_cart[0], goal_pos_cart[1] - goal_pos_prev_cart[1], goal_pos_cart[2] - goal_pos_prev_cart[2]]
        goal_pos_prev_cart = goal_pos_cart
        
        motor_pos_step = motor_pos[index-1]
        motor_pos_delta = [motor_pos_step[0] - motor_pos_prev[0], motor_pos_step[1] - motor_pos_prev[1], motor_pos_step[2] - motor_pos_prev[2], motor_pos_step[3] - motor_pos_prev[3], motor_pos_step[4] - motor_pos_prev[4], motor_pos_step[5] - motor_pos_prev[5], motor_pos_step[6] - motor_pos_prev[6] ]
        motor_pos_prev = motor_pos_step
        
        trial.append([i, pos, goal_delta, goal_pos, motor_pos_center, motor_pos_delta, motor_pos_step, pos_cart, goal_delta_cart, goal_pos_cart ])
    #print trial    
    return trial