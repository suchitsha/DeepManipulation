# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 670, -1, -1),
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


def train_learner(name,fname):

    # read data
    objs = []
    f = open(fname,"rb")
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    #print objslearn_motor_pos
    el_X = []
    el_y = []
    for i in range( len(objs) ):
        # to train over x= time,pos --> positions y= goal_delta --> positions delta
        '''
        temp_x = [objs[i][0],objs[i][1][0],objs[i][1][1],objs[i][1][2]]
        el_X.append(temp_x)
        temp_y = [objs[i][2][0], objs[i][2][1], objs[i][2][2]]
        el_y.append(temp_y)
        #'''
        # to train over x= time,pos --> positions y= motor_pos_delta --> motor positions delta
        #'''
        temp_x = [objs[i][0],objs[i][1][0],objs[i][1][1],objs[i][1][2]]
        el_X.append(temp_x)
        temp_y = [objs[i][5][0], objs[i][5][1], objs[i][5][2], objs[i][5][3], objs[i][5][4] , objs[i][5][5], objs[i][5][6]]
        el_y.append(temp_y)
        #'''
    X = np.array(el_X)
    y = np.array(el_y)    
    #print "X:",X[0]
    #print "-----------------------"
    #print "y:",y[0]
    clfs = []
    if name == 'svm':
        print 'using svm'        
        '''
        X = [[0, 0, 0], [2, 2, 2]]
        y = [0.5, 2.5]
        for i in range(app.num_of_joints):         
            clf = svm(X,y)
            clfs.append(clf)
        '''            
        clf = svm(X,y)
        clfs.append(clf)
    elif name=="lin":
        print "using linear regression"
        clf = lin(X,y)
        clfs.append(clf)
    elif name=="lin_bayesian_ridge":
        print "using bayesian ridge"
        clf = lin_bayesian_ridge(X,y)
        clfs.append(clf)
    elif name=="lin_lasso":
        print "using lasso"
        clf = lin_lasso(X,y)
        clfs.append(clf)
    elif name=="multi":
        print "using multi"
        clf = multi(X,y)
        clfs.append(clf)
    elif name=="rf":
        print "using random forest"
        clf = rf(X,y)
        clfs.append(clf)
    else:
        print 'model not available'       
    return clfs




def eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius):
    s_coord = cartToSpher(pos_x,pos_y,pos_z)
    sum_d = 0.0
    sum_theta = 0.0
    sum_phi = 0.0
    
    #for error
    sum_x = pos_x
    sum_y = pos_y
    sum_z = pos_z
    sum_y2 = pos_y
    sum_z2 = pos_z
    total_error = 0.0

    # create SimpleActionClient object
    #client = actionlib.SimpleActionClient('MaterialOrder', common.msg.MaterialOrderAction)

    # wait for action server to start up
    #client.wait_for_server()
    
    print "evaluating for:", len(clfs) ,"learners"
    print clfs
    for i in range(num_points+1):
        for j in range(len(clfs)):
            result = clfs[j].predict([[i, s_coord[0] , s_coord[1], s_coord[2] ]])
            #print 'result:' , result
            sum_d += result[0][0]/float(len(clfs)) # divide by n as we are adding terms from n classifiers
            sum_theta += result[0][1]/float(len(clfs))
            sum_phi += result[0][2]/float(len(clfs))

        goal_s_x = s_coord[0] + sum_d
        goal_s_y = s_coord[1] + sum_theta
        goal_s_z = s_coord[2] + sum_phi          
        goal_cart = spherToCart(goal_s_x, goal_s_y, goal_s_z)
        #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
        #print 'goal:', goal_cart 
        '''
        goal = common.msg.MaterialOrderGoal()
        goal.order_id = i
        goal.robot_id = 1
        goal.target_pos_x = goal_cart[0]
        goal.target_pos_y = goal_cart[1]
        goal.target_pos_z = goal_cart[2]
        goal.target_orient_x = 0.0
        goal.target_orient_y = 0.0
        goal.target_orient_z = 0.0
        goal.target_orient_w = 0.0
            
        #rospy.loginfo("DISPATCHER: Order %i: Sending to Robot %i" % (goal.order_id, goal.robot_id))
        client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(order_timeout) )
        print "motion controller:", client.get_result()
        #time.sleep(1)
        '''
        #calculate error
        angle = 2.0*math.pi*(i)/num_points
        exp_x= sum_x + math.cos(angle)*radius
        
        '''if i%2:
            sum_y += 1/10.
        else:
            sum_y -= 1/10. '''
        exp_y = sum_y + math.sin(angle)*radius
        if i%2:
            sum_z += 1/10.
        else:
            sum_z -= 1/10.     
        exp_z = sum_z  #+ math.sin(angle)*radius
        #second orientation
        if i%2:
            sum_y2 += 1/10.
        else:
            sum_y2 -= 1/10. 
        exp_y2 = sum_y2 #+ math.sin(angle)*radius
        '''if i%2:
            sum_z2 += 1/10.
        else:
            sum_z2 -= 1/10. '''     
        exp_z2 = sum_z  + math.sin(angle)*radius
        
        error = calc_error( goal_cart, exp_x, (exp_y + exp_y2)/2., (exp_z + exp_z2)/2 )
        print "error:", error
        total_error += error
    
    print "total_error:", total_error/num_points

    return

       
def svm(X,y):
    clf = SVR(C=1.0, epsilon=0.2)
    #clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    #kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
    clf.fit(X,y)
    return clf

def lin(X,y):
    clf = linear_model.LinearRegression()
    clf.fit(X,y)
    return clf

def lin_bayesian_ridge(X,y):
    clf = linear_model.BayesianRidge()
    clf.fit(X,y)
    return clf
    
def lin_lasso(X,y):
    clf = linear_model.Lasso()
    clf.fit(X,y)
    return clf    

#http://stackoverflow.com/questions/21556623/regression-with-multi-dimensional-targets    
def multi(X,y):
    #clf = MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=-1).fit(X, y)
    clf = MultiOutputRegressor(GradientBoostingRegressor(loss='lad' ), n_jobs=-1).fit(X, y)
    #clf = MultiOutputRegressor(GradientBoostingRegressor(loss='lad',n_estimators=100,max_depth=10 ), n_jobs=-1).fit(X, y)
    #clf = MultiOutputRegressor(RandomForestRegressor(max_depth=30, random_state=0)).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor() )).fit(X, y)
    #clf = MultiOutputRegressor( SVR(C=1.0, epsilon=0.2, degree=10) ).fit(X, y)
    return clf    

def rf(X,y):
    clf = RandomForestRegressor(max_depth=30, random_state=2)
    clf.fit(X,y)
    return clf   
     
def train_dnn(training_set):     
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
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[1024, 512, 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],#256,256,256,256,256,256,256,256,256,256,256,256,256],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=dim,
                                            #optimizer=tf.train.ProximalAdagradOptimizer(
                                            #learning_rate=0.001,
                                            #l1_regularization_strength=0.001),
                                            # motor pos small data large repetitions cart coord
                                            #model_dir="/home_local/shar_sc/learn_motion_models_small_cart_motorpos"
                                            
                                            
                                            
                                            
                                            #real robot data
                                            #model_dir="/home_local/shar_sc/learn_motion_models_scrub1" # scrub
                                            #model_dir="/home_local/shar_sc/learn_motion_models_accumulation2" # accum
                                            model_dir="/home_local/shar_sc/learn_motion_models_accumulation5"# 4" # accum4 with orientation
                                            
                                            
                                            
                                            # cart pos small data large repetitions cart coord
                                            #model_dir="/home_local/shar_sc/learn_motion_models_small_cart_cartpos"
                                            
                                            #for motor pos circle for cart coord
                                            #model_dir="/home_local/shar_sc/learn_motion_models"
                                            
                                            # for motor pos circle works fine for spherical coordinate
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp",
                                            #best for cart pos circle
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp_cartesian",
                                            #cart_pos testing 
                                            #model_dir="/home_local/shar_sc/learn_motion_models_temp_cartesian2",
                                            )
    # Fit
                                            
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=1)
    return regressor

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
    if(not dnn_prerecorded):
        #clfs = train_learner("svm") # not for 3 dim Y
        #clfs = train_learner("lin")
        #clfs = train_learner("lin_bayesian_ridge") # not for 3 dim Y
        #clfs = train_learner("lin_lasso")
        print "training"
        print sklearn.__version__
        #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_100_10.p")
        clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data1_5000_10.p")
        #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_8_20_20.p")
        #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_88_20_20.p")
        #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_88_1000_20.p")
        #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_8888_500_60.p")
        #clfs = train_learner("rf")
        
        #two learners at a time
        #clfs.extend(train_learner("multi","data5.p"))

    else:
        # for dnn
        #train
        training_set = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_accum5.p",app.output)#data_rec_accum2.p",app.output)#scrub1.p",app.output)#accum2.p",app.output)#data_rec_circle60.p",app.output)#data2_21300_10.p",app.output)
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
        app.rave.set_config_dict(app.start_cfg)
        
        ''' # for single random point
        r = random.randint(len(app.particles[0]))
        print r
        print app.particles
        x_part = app.particles[0][r] #+ 0.2 #max(app.particles[0])
        y_part = app.particles[1][r] #- 0.15 #min(app.particles[1])
        z_part = 0.01 #TODO 

        roi_frame = odb_utils.float16_to_array(app.wsr.object_store["tray"]["toolframe"])
        #of_frame = dot(dot(app.rave.get_frame("tray"), roi_frame),txyz(x_part,y_part,z_part))
        of_frame = dot(dot(app.rave.get_frame("tray"), roi_frame), dot(txyz(x_part,y_part,z_part), rotz(pi)) )
        hand_frame = dot(of_frame, app.grasp_frame)                
        app.rave.add_coord("ofse", of_frame,"small")
        #app.rave.add_coord("hand", hand_frame)
        
        #initial = app.rave.get_config_dict()
        #q_dict_3 = app.find_rotational_ik(hand_frame, 1)
        #armconfig = app.rave.find_rotational_ik("right_arm", app.grasp_frame, of_frame, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
        armconfig = app.rave.find_rotational_ik("right_arm", app.grasp_frame, of_frame, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
        q_dict_3 = app.rave.get_config_dict()
        q_dict_3["right_arm"] = armconfig        

        if q_dict_3 is None:
            print "cannot find IK"
            return       
        app.rave.set_config_dict(q_dict_3)               
        app.evaluate_data_demo_accum(regressor, num_points)

        '''
        
        x_pmax = max(app.particles[0])
        y_pmin = min(app.particles[1])
        margin_top = 0.07#9#TODO
        for pt in range(len(app.boundary_points.vertices)):
            x_part = app.particles[0][app.boundary_points.vertices[pt]] -margin_top #app.particles[0][app.boundary_points.simplices[pt][0]]
            y_part = app.particles[1][app.boundary_points.vertices[pt]] #app.particles[1][app.boundary_points.simplices[pt][1]]
            z_part = 0.01 #TODO 
            
            if  (x_part > (x_pmax/2 -margin_top) ): # and (y_part < y_pmin/2 ) ):
                roi_frame = odb_utils.float16_to_array(app.wsr.object_store["tray"]["toolframe"])
                #of_frame = dot(dot(app.rave.get_frame("tray"), roi_frame),txyz(x_part,y_part,z_part))
                of_frame = dot(dot(app.rave.get_frame("tray"), roi_frame), dot(txyz(x_part,y_part,z_part), rotz(pi)) )
                hand_frame = dot(of_frame, app.grasp_frame)  
                name_coord = "ofse" + str(pt)              
                app.rave.add_coord(name_coord, of_frame,"small")
                #app.rave.add_coord("hand", hand_frame)
                
                #initial = app.rave.get_config_dict()
                #q_dict_3 = app.find_rotational_ik(hand_frame, 1)
                #armconfig = app.rave.find_rotational_ik("right_arm", app.grasp_frame, of_frame, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
                armconfig = app.rave.find_rotational_ik("right_arm", app.grasp_frame, of_frame, [0, 0, 1], 0, 360, 10, best=True, check_env_collisions=False)
                q_dict_3 = app.rave.get_config_dict()
                q_dict_3["right_arm"] = armconfig        

                if q_dict_3 is None:
                    print "cannot find IK"
                    return       
                app.rave.set_config_dict(q_dict_3)               
                app.evaluate_data_demo_accum(regressor, num_points)
        return
        
        
        
    if (dnn_prerecorded):
        #q_dict = app.rave.get_config_dict()    
        t = app.rave.get_manip_frame("right_arm")
        data = []
        f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_accum5.p","rb")#accum2.p","rb")#scrub1.p","rb")#accum2.p","rb")#data_200_10.p","rb")
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
                #traj = app.calculate_circle_positions(num_points, radius):
                start = (i*num_points)+i
                end = (i*num_points+num_points)+i+1
                trial = data[start:end]   
                print len(trial), start, end 
                app.evaluate_data_from_file_dnn_poly(regressor, trial, i)
                # multiple learners
                #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
                sleep(3)
        return    
    if (prerecorded):
        #q_dict = app.rave.get_config_dict()    
        t = app.rave.get_manip_frame("right_arm")
        data = []
        f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_200_101.p","rb")
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
  
        for i in range(iterations):
            app.rave.clear_graphic_handles("*")
            print "prerecorded iteration:",i+1
            #traj = app.calculate_circle_positions(num_points, radius):
            start = (i*num_points)+i
            end = (i*num_points+num_points)+i+1
            trial = data[start:end]   
            print len(trial), start, end 
            app.evaluate_data_from_file(clfs[0], trial, i)
            # multiple learners
            #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
            sleep(1)
        return
    for i in range(iterations):
        print "iteration:",i+1
        pos_x = random.random()*(max_x - margin - min_x - margin) + (min_x + margin)
        pos_y = random.random()*(max_y - margin - min_y - margin) + (min_y + margin)
        pos_z = random.random()*(max_z - margin - min_z - margin) + (min_z + margin)
        #pos_x = 1.0 pos_y = -0.1 pos_z = 1.0

        app.evaluate_random_point(clfs[0], pos_x, pos_y, pos_z, num_points, radius)
        # multiple learners
        #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
        time.sleep(3)    
    return
