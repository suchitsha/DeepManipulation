# {'exits': ['out'],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (380, 400, -1, -1),
#  'transitions': [('out', 'circle_dnn_poly_multipl_lrnrs_torso')]}
### end of header
import time
import math
import random
import cPickle as pickle
import numpy as np
from pyutils.matrix import *
import itertools
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def init(self):
    
    app.initial_run = True
    # steps to partially train secondary network
    app.num_steps = 100
    app.cost = 0.0 # a big value for beginning
    app.sum_error = 0.0
    app.sum_error_sec = 0.0
    app.repetitions = 0
    # 0 - cartisean pos
    # 1 - motor pos
    app.output = 1
    #True for cartesian coordinates
    app.noSpherical = True
    app.simple = True # simple version with less features
    
    app.FEATURES = ["time", "x", "y", "z"]
    app.COLUMNS = ["time", "x", "y", "z","out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9"]        
    app.LABEL = ["out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9"]
    if app.output == 0:    
        app.COLUMNS = ["time", "x", "y", "z","out_x", "out_y", "out_z"]
        app.LABEL = ["out_x","out_y","out_z"]

    # for secondary learner
    if app.simple == True:
        app.FEATURES_sec = ["time", "x", "y", "z","e_x","e_y","e_z"]
        app.COLUMNS_sec = ["time", "x", "y", "z", "e_x","e_y","e_z", "out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9"]        
        app.LABEL_sec = ["out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9"]
        if app.output == 0:    
            app.FEATURES_sec = ["time", "x", "y", "z","e_x","e_y","e_z"]
            app.COLUMNS_sec = ["time", "x", "y", "z","e_x","e_y","e_z", "out_x", "out_y", "out_z"]
            app.LABEL_sec = ["out_x","out_y","out_z"]        
    else: 
        app.FEATURES_sec = ["time", "x", "y", "z","r1","r2","r3","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","e_x","e_y","e_z"]
        app.COLUMNS_sec = ["time", "x", "y", "z","r1","r2","r3","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9", "e_x","e_y","e_z", "out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9"]        
        app.LABEL_sec = ["out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9"]
        if app.output == 0:    
            app.FEATURES_sec = ["time", "x", "y", "z","r1","r2","r3","p0","p1","p2","e_x","e_y","e_z"]
            app.COLUMNS_sec = ["time", "x", "y", "z","r1","r2","r3","p0","p1","p2", "e_x","e_y","e_z", "out_x", "out_y", "out_z"]
            app.LABEL_sec = ["out_x","out_y","out_z"]        
        
    tf.logging.set_verbosity(tf.logging.INFO)
    app.LEARNING_RATE = 0.01# .001 default
    pass

def execute(self):
    print "execute of %s called!" % self.name
    learnMotion()
    return dict(exit="out")

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

def train_dnn(training_set):     
    print "Training Regressor:"
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in app.FEATURES]
    if app.output == 0:
        app.dim = 3
    else:
        app.dim = 10  
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[1024, 512, 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],#256,256,256,256,256,256,256,256,256,256,256,256,256],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=app.dim,
                                            #optimizer=tf.train.ProximalAdagradOptimizer(
                                            #learning_rate=0.001,
                                            #l1_regularization_strength=0.001),
                                            # motor pos small data large repeatitions cart coord
                                            #model_dir="/home_local/shar_sc/learn_motion_models_small_cart_motorpos"
                                            
                                            #for motor pos circle for cart coord
                                            model_dir="/home_local/shar_sc/learn_motion_models_wipe1"
                                            
                                            # for motor pos circle works fine for spherical coordinate
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp",
                                            #best for cart pos circle
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp_cartesian",
                                            #cart_pos testing 
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp_cartesian2",
                                            )

    # Fit
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=40)
    return regressor

def model_fn(features, targets, mode, params):
    if app.output == 0:
        app.dim = 3
    else:
        app.dim = 10  
    """Model function for Estimator."""
    # Connect the first hidden layer to input layer
    # (features) with relu activation    
    first_hidden_layer = tf.contrib.layers.relu(features, 100)
    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 100)
    t_hidden_layer = tf.contrib.layers.relu(second_hidden_layer, 100)
    f_hidden_layer = tf.contrib.layers.relu(t_hidden_layer, 50)
    fv_hidden_layer = tf.contrib.layers.relu(f_hidden_layer, 50)
    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(fv_hidden_layer, app.dim)
    # Reshape output layer to 1-dim Tensor to return predictions
    #predictions = tf.reshape(output_layer, [-1])
    #predictions_dict = {"pred": predictions}

    # Calculate loss using mean squared error
    loss = tf.Variable(app.cost, name="loss", dtype=tf.float64) # tf.losses.mean_squared_error(targets, predictions)
    
    '''
    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
        tf.cast(targets, tf.float64), predictions)
    }
    '''
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="SGD")

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=output_layer, #predictions_dict,
        loss=loss,
        train_op=train_op)#, eval_metric_ops=eval_metric_ops)


def create_dnn_secondary(training_set, cost):     
    print "Training Regressor:"
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in app.FEATURES]
                  #for k in app.FEATURES_sec]
    if app.output == 0:
        app.dim = 3
    else:
        app.dim = 10  
    
    optmzr = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    opt_op = optmzr.minimize(cost)
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor_s = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[1024, 512, 256,256,256],#,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],#256,256,256,256,256,256,256,256,256,256,256,256,256],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=app.dim,
                                            #optimizer=opt_op,
                                            optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001,
                                            l1_regularization_strength=0.001).minimize(cost),
                                            # motor pos small data large repetitions cart coord
                                            #model_dir="/home_local/shar_sc/learn_motion_models_small_cart_motorpos"
                                            # motor pos secondary net cart coord
                                            model_dir="/home_local/shar_sc/learn_motion_models_secondary_cart_motorpos"
                                            
                                            #for motor pos circle for cart coord
                                            ##model_dir="/home_local/shar_sc/learn_motion_models"
                                            
                                            # for motor pos circle works fine for spherical coordinate
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp",
                                            #best for cart pos circle
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp_cartesian",
                                            #cart_pos testing 
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp_cartesian2",
                                            )
    # Fit                                            
    #regressor_s.partial_fit(input_fn=lambda: input_fn_sec(training_set), steps=app.num_steps)
    regressor_s.partial_fit(input_fn=lambda: input_fn(training_set), steps=app.num_steps)
    return regressor_s
    
def partial_fit_regressor(regressor_s, data, num_steps):
    # Fit
    regressor_s.partial_fit(input_fn=lambda: input_fn(data), steps=num_steps)
    #regressor_s.partial_fit(input_fn=lambda: input_fn_sec(data), steps=num_steps)
    return regressor_s

def process_data(file_name,output=0):
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
                if app.noSpherical:
                    list_x.append(line[7][0])
                    list_y.append(line[7][1])
                    list_z.append(line[7][2])                                  
                    data_final_pos_x.append(line[8][0])
                    data_final_pos_y.append(line[8][1])
                    data_final_pos_z.append(line[8][2])                        
                else:
                    list_x.append(line[1][0])
                    list_y.append(line[1][1])
                    list_z.append(line[1][2])
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
        data_torso_1 = []
        data_torso_2 = []
        data_torso_3 = []
                        
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
                data_torso_1.append(line[11][0])                                                                                                
                data_torso_2.append(line[11][1])                                                                                                
                data_torso_3.append(line[11][2])                                                                                                                                
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
             'out_6': data_motor_pos_6,
             'out_7': data_torso_1,
             'out_8': data_torso_2,
             'out_9': data_torso_3                                                                                                                                                            
             } 
        return pd.DataFrame(d)
    
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in app.FEATURES}
    labels = tf.constant(data_set[app.LABEL].values)
    return feature_cols, labels
        
def input_fn_sec(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in app.FEATURES_sec}
    labels = tf.constant(data_set[app.LABEL_sec].values)
    return feature_cols, labels

def learnMotion():
    tf.logging.set_verbosity(tf.logging.INFO)   
    #generate trajectory
    num_points = 10      #10 # points on circle
    skip = 0
    iterations = 1#25#112      # trajectories to be generated

    #train
    training_set = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec2.p",app.output)#data2_8000_10.p",app.output)
    regressor = train_dnn(training_set) 
    
    ''' To test and evaluate dnn without executing trajectory
    test_set_list = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_88_20_20.p",app.output)
    prediction_set_list = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_88_20_20.p",app.output)
    # Score accuracy
    ev = regressor_list[0].evaluate(input_fn=lambda: input_fn(test_set_list[0]), steps=1)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    # Print out predictions
    y = regressor_list[0].predict(input_fn=lambda: input_fn(prediction_set_list[0]))
    # .predict() returns an iterator; convert to a list and print predictions
    predictions = list(itertools.islice(y, 100))
    print("Predictions: {}".format(str(predictions)))
    #'''
    
    print "evaluating"
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
    
    #q_dict = app.rave.get_config_dict()    
    t = app.rave.get_manip_frame("right_arm")
    data = []
    f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec2.p","rb")    
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

    for i in range(iterations+skip):
        if(i < skip):
            print "skipping ",i,"th data for replay"
        else:
            app.rave.clear_graphic_handles("*")
            print "dnn prerecorded iteration:",i+1
            start = (i*num_points)+i
            end = (i*num_points+num_points)+i+1
            trial = data[start:end]   
            print len(trial), start, end 
            evaluate_data_from_file_dnn_poly(regressor, trial, i)    
            sleep(0.1)
    if (app.repetitions!=0):
        err = app.sum_error/app.repetitions
        print "average error over all iterations:", err, app.repetitions
        err_sec = app.sum_error_sec/app.repetitions
        print "average secondory error over all iterations:", err_sec, app.repetitions
    else:
        print "no iteration executed"    
    return

def calc_error(g_cart, x,y,z):
    error = math.sqrt( math.pow((g_cart[0] - x),2) + math.pow((g_cart[1] - y),2) + math.pow((g_cart[2] - z),2) )
    #print "errorx:", g_cart[0] - x , "errory:", g_cart[1] - y , "errorz:", g_cart[2] - z
    return error 
        
def evaluate_data_from_file_dnn_poly(regressor, data, name=""):
    num_points = len(data)
        
    #for center frame, including spherical coords, and motor pos at center
    app.idle_cfg["right_arm"] = data[0][4]
    app.idle_cfg["torso"] = data[0][10]
    app.rave.set_config_dict(app.idle_cfg)
    center_frame = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("goal", center_frame,"small")
    s_coord = cartToSpher(*center_frame[0:3,3])
    #c_coord = [center_frame[0,3], center_frame[1,3], center_frame[2,3]]
    c_coord = [center_frame[0,3]-0.58, center_frame[1,3]+.29, center_frame[2,3]-0.99]
    #print "c_coord:",c_coord
    
    q_dict = app.rave.get_config_dict()  
    motor_pos_center = q_dict["right_arm"]
    torso_center = q_dict["torso"]
    
    total_error = 0.0
    total_error_sec = 0.0    
    points = []
    points_sec = []
    points_expected = []
    points_failure = []
    sum_motor_pos = [0.,0.,0.,0.,0.,0.,0.]
    sum_motor_pos_sec = [0.,0.,0.,0.,0.,0.,0.]
    sum_torso = [0.,0.,0.]
    sum_torso_sec = [0.,0.,0.]
    sum_motor_failure = []    
       
    #what was recorded:
    for data_point in data[0:-1]:
        app.idle_cfg["right_arm"] = data_point[6]#[-1][6]#  data_point[-1]
        app.idle_cfg["torso"] = data_point[12]
        app.rave.set_config_dict(app.idle_cfg)
        expected = app.rave.get_manip_frame("right_arm")
        points_expected.append(expected[0:3,3])
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        app.rave.draw_line("line_expected%s" % name, points_expected, [0,1,0], 4)
        
        '''
        #introduce error
        failure = app.rave.get_manip_frame("right_arm")
        failure = dot(txyz(0, 0, 0.1), failure)
        failure_cfg = app.ikine.dual_ik(None, failure, initial=app.idle_cfg)
        sum_motor_failure.append(app.rave.dict_to_configlist(failure_cfg) - app.rave.dict_to_configlist(app.idle_cfg))
        #app.rave.set_config_dict(failure_cfg)
        points_failure.append(failure[0:3,3]) 
        app.rave.draw_line("line_failure%s" % name, points_failure, [1,0,0], 4)
        #sleep(0.5)
        '''
        
    points_expected.append(points_expected[0])

    sum_d = 0.0
    sum_theta = 0.0
    sum_phi = 0.0
    
    #result of estimation
    for i in xrange(num_points):
        result = []
        list_x = []
        list_y = []
        list_z = [] 
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
                 'out_6': 0.0,          
                 'out_7': 0.0,
                 'out_8': 0.0,
                 'out_9': 0.0                                                                                               
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
                offset[0,3] = goal_cart[0]# + 0.65
                offset[1,3] = goal_cart[1]
                offset[2,3] = goal_cart[2]# + 0.90
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
            #'''
            sum_motor_pos[0] += predictions[0][0] #+0.01 #result[0]
            sum_motor_pos[1] += predictions[0][1] #+0.1 # induce error for second network to correct
            sum_motor_pos[2] += predictions[0][2] #+0.01
            sum_motor_pos[3] += predictions[0][3] #+0.01
            sum_motor_pos[4] += predictions[0][4] #- 0.4
            sum_motor_pos[5] += predictions[0][5] #- 0.4
            sum_motor_pos[6] += predictions[0][6] #- 0.4
            sum_torso[0]     += predictions[0][7]
            sum_torso[1]     += predictions[0][8]
            sum_torso[2]     += predictions[0][9]
            #'''
            
            #sum_motor_pos[:7] += predictions[0][0:7] + sum_motor_failure[i][3:10]
            #sum_torso[:3] += predictions[0][7:] + sum_motor_failure[i][:3]

                        
            #print "smp: " ,sum_motor_pos
            estimated_pos = [0.,0.,0.,0.,0.,0.,0.]
            estimated_pos_torso = [0.,0.,0.]
            estimated_pos = motor_pos_center + sum_motor_pos
            estimated_pos_torso = torso_center + sum_torso
            
            #move robot to estimated pose
            cdict = app.rave.get_config_dict()
            cdict["right_arm"] = estimated_pos
            cdict["torso"] = estimated_pos_torso
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
        
        
        #TODO
        continue
        
        
        
        
        #data for other network
        e_x = reached_frame[0:3,3][0] - points_expected[i][0]
        e_y = reached_frame[0:3,3][1] - points_expected[i][1]
        e_z = reached_frame[0:3,3][2] - points_expected[i][2]   
        list_x1 = []   
        list_y1 = []  
        list_z1 = []            
        if app.noSpherical:
            list_x1 = [c_coord[0]]
            list_y1 = [c_coord[1]]
            list_z1 = [c_coord[2]]                                
        else:
            list_x1 = [s_coord[0]]
            list_y1 = [s_coord[1]]
            list_z1 = [s_coord[2]]
        if app.output == 0:
            if app.simple == True:
                d_sec = {'time' : i,        # time and center of trajectory
                     'x' : list_x1,
                     'y' : list_y1,
                     'z' : list_z1,
                     'e_x' : e_x,       # errors from current predictions
                     'e_y' : e_y,
                     'e_z' : e_z,
                     'out_x': 0.0, # no values
                     'out_y': 0.0,
                     'out_z': 0.0
                     }
            else:
                d_sec = {'time' : i,        # time and center of trajectory
                     'x' : list_x1,
                     'y' : list_y1,
                     'z' : list_z1,
                     'r1' : reached_frame[0:3,3][0],     # current position of end effector
                     'r2' : reached_frame[0:3,3][1],
                     'r3' : reached_frame[0:3,3][2],                                 
                     'p0' : predictions[0][0],       # predicted deltas
                     'p1' : predictions[0][1],
                     'p2' : predictions[0][2],
                     'e_x' : e_x,       # errors from current predictions
                     'e_y' : e_y,
                     'e_z' : e_z,
                     'out_x': 0.0, # no values
                     'out_y': 0.0,
                     'out_z': 0.0
                     }
                             
            #x_sec = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0],reached_frame[0:3,3][0],reached_frame[0:3,3][1],reached_frame[0:3,3][2],predictions[0][0],predictions[0][1],predictions[0][2], e_x, e_y, e_z ] ])             
            #y_sec = tf.constant([ [0.0, 0.0, 0.0] ])
        else: 
            if app.simple==True:            
                d_sec = {'time' : i,        # time and center of trajectory
                     'x' : list_x1,
                     'y' : list_y1,
                     'z' : list_z1,
                     'e_x' : e_x,      # errors from current predictions
                     'e_y' : e_y,
                     'e_z' : e_z,
                     'out_0': 0.0, # no values
                     'out_1': 0.0,
                     'out_2': 0.0,
                     'out_3': 0.0,
                     'out_4': 0.0,
                     'out_5': 0.0,
                     'out_6': 0.0,
                     'out_7': 0.0,
                     'out_8': 0.0,
                     'out_9': 0.0                                                                                                                                
                     }
            else:
                d_sec = {'time' : i,        # time and center of trajectory
                     'x' : list_x1,
                     'y' : list_y1,
                     'z' : list_z1,
                     'r1' : reached_frame[0:3,3][0],     # current position of end effector
                     'r2' : reached_frame[0:3,3][1],
                     'r3' : reached_frame[0:3,3][2],                                                   
                     'p0' : predictions[0][0],       # predicted deltas
                     'p1' : predictions[0][1],
                     'p2' : predictions[0][2],
                     'p3' : predictions[0][3],
                     'p4' : predictions[0][4],
                     'p5' : predictions[0][5],
                     'p6' : predictions[0][6],
                     'p7' : predictions[0][7],
                     'p8' : predictions[0][8],
                     'p9' : predictions[0][9],                                                   
                     'e_x' : e_x,      # errors from current predictions
                     'e_y' : e_y,
                     'e_z' : e_z,
                     'out_0': 0.0, # no values
                     'out_1': 0.0,
                     'out_2': 0.0,
                     'out_3': 0.0 ,
                     'out_4': 0.0,
                     'out_5': 0.0,
                     'out_6': 0.0,
                     'out_7': 0.0,
                     'out_8': 0.0,
                     'out_9': 0.0                                                                                                                                
                     }
                                 
            #x_sec = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0],reached_frame[0:3,3][0],reached_frame[0:3,3][1],reached_frame[0:3,3][2],predictions[0][0],predictions[0][1],predictions[0][2],predictions[0][3],predictions[0][4],predictions[0][5],predictions[0][6], e_x, e_y, e_z ] ])                              
            #y_sec = tf.constant([ [0.0,0.0,0.0,0.0,0.0,0.0,0.0] ])
        input_data_sec = pd.DataFrame(d_sec) 
        
        #cost = tf.Variable([e_x,e_y,e_z], name="cost", dtype=tf.float32) #+ current prediction or Y_pred #tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)
        #app.cost = error
        # Set model params
        model_params = {"learning_rate": app.LEARNING_RATE}
        # Instantiate Estimator
        app.nn = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir="/home_local/shar_sc/learn_motion_models_secondary_cart_motorpos_5", params=model_params)
        # directory 2 for simple false , 3 for simple true        
        #TODO write model and read every time for training
        def get_train_inputs():
            if app.simple == True:
                if app.output == 0:
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0], e_x, e_y, e_z ] ])             
                    y = tf.constant([ [0.0, 0.0, 0.0] ]) #([ [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0] ]])
                else:
                    #TODO are all these parameters necessary? yes but only e_x,e_y,e_z should suffice, maybe time, or later sensor inputs
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0], e_x, e_y, e_z ] ])  
                    #TODO should be 10 parameteres here                            
                    #TODO
                    y = tf.constant([ [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] ]) 
            else:
                if app.output == 0:
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0],reached_frame[0:3,3][0],reached_frame[0:3,3][1],reached_frame[0:3,3][2],predictions[0][0],predictions[0][1],predictions[0][2], e_x, e_y, e_z ] ])             
                    y = tf.constant([ [0.0, 0.0, 0.0] ]) #([ [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0] ]])
                else:
                    #TODO are all these parameters necessary? yes but only e_x,e_y,e_z should suffice, maybe time, or later sensor inputs
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0],reached_frame[0:3,3][0],reached_frame[0:3,3][1],reached_frame[0:3,3][2],predictions[0][0],predictions[0][1],predictions[0][2],predictions[0][3],predictions[0][4],predictions[0][5],predictions[0][6],predictions[0][7],predictions[0][8],predictions[0][9], e_x, e_y, e_z ] ])                              
                    #TODO should be 9 parameteres here
                    #TODO
                    y = tf.constant([ [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] ]) 
            '''sess= tf.Session()
            a = sess.run(y)
            print a'''
            return x, y
        
        def get_pred_inputs():
            if app.simple==True: 
                if app.output == 0:
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0], e_x, e_y, e_z ] ])             
                else:
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0], e_x, e_y, e_z ] ])
            else:
                if app.output == 0:
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0],reached_frame[0:3,3][0],reached_frame[0:3,3][1],reached_frame[0:3,3][2],predictions[0][0],predictions[0][1],predictions[0][2], e_x, e_y, e_z ] ])             
                else:
                    x = tf.constant([ [ i,list_x1[0],list_y1[0],list_z1[0],reached_frame[0:3,3][0],reached_frame[0:3,3][1],reached_frame[0:3,3][2],predictions[0][0],predictions[0][1],predictions[0][2],predictions[0][3],predictions[0][4],predictions[0][5],predictions[0][6],predictions[0][7],predictions[0][8],predictions[0][9], e_x, e_y, e_z ] ])            
            return x
        
        # Fit
        #nn.fit(input_fn=get_train_inputs, steps=1000)
        #app.nn.partial_fit(input_fn=get_train_in0.1 puts,steps=1) # train once cause the app.cost is not yet updated

        '''
        ev = app.nn.evaluate(input_fn=get_test_inputs, steps=1)
        print("Loss: %s" % ev["loss"])
        #print("Root Mean Squared Error: %s" % ev["rmse"])
        '''    
        # Print out predictions
        if (app.initial_run == True):
            app.initial_run = False
            app.nn.partial_fit(input_fn=get_train_inputs,steps=1)
                
        result_sec = app.nn.predict(input_fn=get_pred_inputs) #(x=prediction_set.data, as_iterable=True)
        predictions_sec = None
        for r, p in enumerate(result_sec):
            print "Prediction secondary: ", p
            predictions_sec = p
        #print("Predictions sec: {}".format(str(predictions_sec)))
        
        '''
        # train other network
        if (app.initial_run == True):
            #learning_rate = 0.01
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            #app.regressor_sec = create_dnn_secondary(input_data_sec , optimizer)
            app.regressor_sec = create_dnn_secondary(input_data , cost)
            app.initial_run = False
        elif(app.initial_run == False):                    
            #app.regressor_sec = partial_fit_regressor(app.regressor_sec, input_data_sec , app.num_steps)
            app.regressor_sec = partial_fit_regressor(app.regressor_sec, input_data , app.num_steps)
            #predict and print predictions
            y_sec = app.regressor_sec.predict(input_fn=lambda: input_fn(input_data))
            #y_sec = app.regressor_sec.predict(input_fn=lambda: input_fn_sec(input_data_sec))
            pred = list(itertools.islice(y_sec, 1))
            print("Pred: {}".format(str(pred)))
        '''            
        # for motor position learned
        '''
        # for individual trajectories and errors at each point
        sum_motor_pos_sec = [0.,0.,0.,0.,0.,0.,0.] # TODO for now just use the current prediction and not take into account previous perdictions
        sum_motor_pos_sec[0] += predictions_sec[0]
        sum_motor_pos_sec[1] += predictions_sec[1]
        sum_motor_pos_sec[2] += predictions_sec[2]
        sum_motor_pos_sec[3] += predictions_sec[3]
        sum_motor_pos_sec[4] += predictions_sec[4]
        sum_motor_pos_sec[5] += predictions_sec[5]
        sum_motor_pos_sec[6] += predictions_sec[6]

        estimated_p = motor_pos_center + sum_motor_pos
        estimated_pos_sec = estimated_p + sum_motor_pos_sec
        '''
        # for combined trajectories and errors at each point
        
        sum_motor_pos_sec[0] += predictions_sec[0]
        sum_motor_pos_sec[1] += predictions_sec[1]
        sum_motor_pos_sec[2] += predictions_sec[2]
        sum_motor_pos_sec[3] += predictions_sec[3]
        sum_motor_pos_sec[4] += predictions_sec[4]
        sum_motor_pos_sec[5] += predictions_sec[5]
        sum_motor_pos_sec[6] += predictions_sec[6]
        sum_torso_sec[0]     += predictions_sec[7]
        sum_torso_sec[1]     += predictions_sec[8]
        sum_torso_sec[2]     += predictions_sec[9]
                        
        estimated_pos_sec = [0.,0.,0.,0.,0.,0.,0.]
        estimated_pos_sec = estimated_pos + sum_motor_pos_sec #motor_pos_center + sum_motor_pos_sec
        estimated_tor_sec = [0.,0.,0.]
        print "sum_torso_sec", sum_torso_sec
        estimated_tor_sec = estimated_pos_torso + sum_torso_sec #torso_center + sum_torso_sec
        #TODO another problem here , solved
        #'''
        



        #move robot to estimated pose
        cdict_sec = app.rave.get_config_dict()
        cdict_sec["right_arm"] = estimated_pos_sec
        #TODO use torso too
        #cdict_sec["torso"] = estimated_tor_sec
        app.rave.set_config_dict(cdict_sec)
              
        #print result
        reached_frame = app.rave.get_manip_frame("right_arm")
        points_sec.append(reached_frame[0:3,3])
        app.rave.draw_line("line", points_sec, [0,1,1], 4)
        #print "reached: ",reached_frame[0:3,3] 
        #print "expected: ",points_expected[i]
        #error assumption
        error_sec = calc_error(points_sec[i], points_expected[i][0], points_expected[i][1], points_expected[i][2])
        print "error_sec:", error_sec  
        #TODO this is actually cost for previous step not current step, correct this
        
        #TODO error is distance from table and not from last trajectory
        app.cost = error_sec*100.0  

        #train again, many times as the cost for this input is known        
        app.nn.partial_fit(input_fn=get_train_inputs,steps=1)

        #add errors        
        total_error += error
        total_error_sec += error_sec            
        
    print "total_error:", total_error/num_points
    print "total_error_sec:", total_error_sec/num_points    
    app.sum_error = app.sum_error + total_error/num_points
    app.sum_error_sec = app.sum_error_sec + total_error_sec/num_points    
    app.repetitions = app.repetitions + 1
    return
