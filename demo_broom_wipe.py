# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (250, 670, -1, -1),
#  'transitions': []}
### end of header
import time
import math
import random
import cPickle as pickle
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from pyutils.matrix import *
import itertools
import pandas as pd
import tensorflow as tf
from odb_interface import odb_utils


def init(self):

    # 0 - cartisean pos
    # 1 - motor pos
    app.output = 1
    #True for cartesian coordinates
    app.noSpherical = True
    
    app.num_steps = 1# 20000
    
    app.FEATURES = ["time", "x", "y", "z"]
    app.COLUMNS = ["time", "x", "y", "z","out_0","out_1","out_2","out_3","out_4","out_5","out_6"]        
    app.LABEL = ["out_0","out_1","out_2","out_3","out_4","out_5","out_6"]
    if app.output == 0:    
        app.COLUMNS = ["time", "x", "y", "z","out_x", "out_y", "out_z"]
        app.LABEL = ["out_x","out_y","out_z"]
    pass

def execute(self):
    print "execute of %s called!" % self.name
    learnMotion()
    return

     
def train_dnnL(training_set):     
    print "Training Regressor:"
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in app.FEATURES]
    #print "feature", feature_cols
    if app.output == 0:
        dim = 3
    else:
        dim = 7  
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressorL = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[1024, 512, 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],#256,256,256,256,256,256,256,256,256,256,256,256,256],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=dim,
                                            #optimizer=tf.train.ProximalAdagradOptimizer(
                                            #learning_rate=0.001,
                                            #l1_regularization_strength=0.001),
                                                                                     
                                            #real robot data
                                            model_dir="/home_local/shar_sc/learn_motion_models_broomL"
                                            )
    # Fit
    regressorL.fit(input_fn=lambda: input_fn(training_set), steps=app.num_steps)
    return regressorL

def train_dnnR(training_set):     
    print "Training Regressor:"
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in app.FEATURES]
    #print "feature", feature_cols
    if app.output == 0:
        dim = 3
    else:
        dim = 7  
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressorR = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[1024, 512, 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],#256,256,256,256,256,256,256,256,256,256,256,256,256],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=dim,
                                            #optimizer=tf.train.ProximalAdagradOptimizer(
                                            #learning_rate=0.001,
                                            #l1_regularization_strength=0.001),
                                                                                     
                                            #real robot data
                                            model_dir="/home_local/shar_sc/learn_motion_models_broomR"
                                            )
    # Fit
    regressorR.fit(input_fn=lambda: input_fn(training_set), steps=app.num_steps)
    return regressorR


def process_data(file_name,output=0):
    #data_pos = []
    data_final_pos_x = []
    data_final_pos_y = []
    data_final_pos_z = []    
    list_x = []
    list_y = []
    list_z = []  
    time = []
    if(output == 0):
        f = open(file_name,"rb")
        while 1:
            try:
                line = pickle.load(f)
                time.append(line[0])
                '''
                list_x.append(line[1][0])
                list_y.append(line[1][1])
                list_z.append(line[1][2])
                '''
                if app.noSpherical:
                    #'''
                    list_x.append(line[7][0])
                    list_y.append(line[7][1])
                    list_z.append(line[7][2])                                  
                    #'''                        
                    data_final_pos_x.append(line[8][0])
                    data_final_pos_y.append(line[8][1])
                    data_final_pos_z.append(line[8][2])                        
                else:
                    #'''
                    list_x.append(line[1][0])
                    list_y.append(line[1][1])
                    list_z.append(line[1][2])
                    #'''
                    data_final_pos_x.append(line[2][0])
                    data_final_pos_y.append(line[2][1])
                    data_final_pos_z.append(line[2][2])                        
            except EOFError:
                break
        f.close()
        d = {'time' : time,
             'x' : list_x,
             'y' : list_y,
             'z' : list_z,
             'out_x': data_final_pos_x,
             'out_y': data_final_pos_y,
             'out_z': data_final_pos_z
             } 
        return pd.DataFrame(d)   
        
    elif(output==1):
        data_motor_pos_0 = []
        data_motor_pos_1 = []
        data_motor_pos_2 = []
        data_motor_pos_3 = []
        data_motor_pos_4 = []
        data_motor_pos_5 = []
        data_motor_pos_6 = []

        f = open(file_name,"rb")
        while 1:
            try:
                line = pickle.load(f)
                time.append(line[0])
                if app.noSpherical:
                    #'''
                    list_x.append(line[7][0])
                    list_y.append(line[7][1])
                    list_z.append(line[7][2])
                    #'''
                else:    
                    list_x.append(line[1][0])
                    list_y.append(line[1][1])
                    list_z.append(line[1][2])                        
                data_motor_pos_0.append(line[5][0])
                data_motor_pos_1.append(line[5][1])
                data_motor_pos_2.append(line[5][2])
                data_motor_pos_3.append(line[5][3])
                data_motor_pos_4.append(line[5][4])
                data_motor_pos_5.append(line[5][5])
                data_motor_pos_6.append(line[5][6])                                                                                
            except EOFError:
                break
        f.close()
        d = {'time' : time,
             'x' : list_x,
             'y' : list_y,
             'z' : list_z,
             'out_0': data_motor_pos_0,
             'out_1': data_motor_pos_1,
             'out_2': data_motor_pos_2,
             'out_3': data_motor_pos_3,
             'out_4': data_motor_pos_4,
             'out_5': data_motor_pos_5,
             'out_6': data_motor_pos_6                                                                              
             } 
        return pd.DataFrame(d)
    
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in app.FEATURES}
    labels = tf.constant(data_set[app.LABEL].values)
    return feature_cols, labels
        
def learnMotion():
    tf.logging.set_verbosity(tf.logging.INFO)   
    #generate trajectory
    max_x = .2#1.5
    min_x = -.2# 0.5
    max_y = .2#1.0
    min_y = -.2#-0.9
    max_z = .2#1.5   print tr
    min_z = -.2
    margin = 0.0
    radius = 0.15
    num_points = 10 #10 # points on circle
    skip = 0
    iterations = 1     # trajectories to be generated
    prerecorded = False
    dnn_prerecorded = True
    demo = True

    # for dnn
    #train
    training_setL = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_broomL.p",app.output)#data_rec_accum2.p",app.output)#scrub1.p",app.output)#accum2.p",app.output)#data_rec_circle60.p",app.output)#data2_21300_10.p",app.output)
    regressorL = train_dnnL(training_setL) 

    training_setR = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_broomR.p",app.output)#data_rec_accum2.p",app.output)#scrub1.p",app.output)#accum2.p",app.output)#data_rec_circle60.p",app.output)#data2_21300_10.p",app.output)
    regressorR = train_dnnR(training_setR) 

    print "evaluating"
    #move sponge perpendicular to table
    temp_cd = app.rave.get_config_dict()
    temp_cd["right_arm"]
    
    '''
    app.idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([0, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0.    ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(app.idle_cfg)
    app.rave.remove_coords("*")
    '''
    #For demo
    if (demo):
        #app.rave.clear_graphic_handles("*")

        dataL = []
        fL = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_broomL.p","rb")#accum2.p","rb")#scrub1.p","rb")#accum2.p","rb")#data_200_10.p","rb")
        while True:
            try:
                dataL.append(pickle.load(fL))
            except EOFError:
                break
                
        dataR = []
        fR = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_broomR.p","rb")#accum2.p","rb")#scrub1.p","rb")#accum2.p","rb")#data_200_10.p","rb")
        while True:
            try:
                dataR.append(pickle.load(fR))
            except EOFError:
                break
        
        cdict = app.rave.get_config_dict()
        cdict["right_arm"] = dataR[20][4] # initial motor pos of first data point
        cdict["left_arm"] = dataL[20][4]
        cdict["torso"] = dataR[20][10]
        #cdict["torso"] = [0, -0.8, 1.5]
        app.rave.set_config_dict(cdict)
        
        #'''
        initial = app.rave.get_config_dict()
        rf = dot(txyz(0.05212, 0.0511, 0.0335), app.rave.get_manip_frame("right_arm"))
        lf = dot(txyz(0.0511, 0.0512, 0.0325), app.rave.get_manip_frame("left_arm"))
        left_armconfig = app.rave.find_rotational_ik("left_arm", eye(4), lf, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
        right_armconfig = app.rave.find_rotational_ik("right_arm", eye(4), rf, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
        cdict = app.rave.get_config_dict()
        cdict["right_arm"] = right_armconfig
        cdict["left_arm"] = left_armconfig
        app.rave.set_config_dict(cdict)
        #'''
        
        #initial = app.rave.get_config_dict()
        #q_dict_3 = app.find_rotational_ik(hand_frame, 1)
        #armconfig = app.rave.find_rotational_ik("right_arm", app.grasp_frame, of_frame, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
        '''
        armconfig = app.rave.find_rotational_ik("right_arm", app.grasp_frame, of_frame, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
        q_dict_3 = app.rave.get_config_dict()
        q_dict_3["right_arm"] = armconfig        

        if q_dict_3 is None:
            print "cannot find IK"
            return       
        app.rave.set_config_dict(q_dict_3)               
        '''
        app.evaluate_data_demo_broom(regressorL, regressorR, num_points)
        return
