# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (280, 720, -1, -1),
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


def init(self):
    app.COLUMNS = ["time", "x", "y", "z","out"]#, "out_y", "out_z"]
    app.FEATURES = ["time", "x", "y", "z"]
    app.LABEL = "out"
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
     
def train_dnn(training_set_list):
    tf.logging.set_verbosity(tf.logging.INFO)        
    regressor_list = []
    for i in range(len(training_set_list)):
        print "Training Regressor:",i
        # Feature cols
        feature_cols = [tf.contrib.layers.real_valued_column(k)
                      for k in app.FEATURES]
        #print "feature", feature_cols

        # Build 2 layer fully connected DNN with 10, 10 units respectively.
        regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                                hidden_units=[1024, 512, 256],
                                                #optimizer=tf.train.ProximalAdagradOptimizer(
                                                #learning_rate=0.1,
                                                #l1_regularization_strength=0.001)
                                                #model_dir="/tmp/learned_model"
                                                )
        # Fit
        regressor.fit(input_fn=lambda: input_fn(training_set_list[i]), steps=1000)#5000)
        regressor_list.append(regressor)
    return regressor_list                                                                

def process_data(file_name,output=0):
    #data_pos = []
    data_final_pos_x = []
    data_final_pos_y = []
    data_final_pos_z = []    
    list_x = []
    list_y = []
    list_z = []  
    time = []
    df = []
    if(output == 0):
        f = open(file_name,"rb")
        while 1:
            try:
                line = pickle.load(f)
                time.append(line[0])
                list_x.append(line[1][0])
                list_y.append(line[1][1])
                list_z.append(line[1][2])                        
                #data_pos.append(line[1])
                data_final_pos_x.append(line[2][0])
                data_final_pos_y.append(line[2][1])
                data_final_pos_z.append(line[2][2])                        
                #data_motor_pos.append(line[5])
            except EOFError:
                break
        f.close()
        d = {'time' : time,
             'x' : list_x,
             'y' : list_y,
             'z' : list_z,
             'out': data_final_pos_x
             } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_final_pos_y
         } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_final_pos_z
         } 
        df.append(pd.DataFrame(d))
        return df   
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
                list_x.append(line[1][0])
                list_y.append(line[1][1])
                list_z.append(line[1][2])                        
                #data_pos.append(line[1])
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
             'out': data_motor_pos_0
             } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_motor_pos_1
         } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_motor_pos_2
         } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
             'x' : list_x,
             'y' : list_y,
             'z' : list_z,
             'out': data_motor_pos_3
             } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_motor_pos_4
         } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_motor_pos_5
         } 
        df.append(pd.DataFrame(d))
        d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out': data_motor_pos_6
         } 
        df.append(pd.DataFrame(d))
        return df   
    
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in app.FEATURES}
    labels = tf.constant(data_set[app.LABEL].values)
    return feature_cols, labels
        
def learnMotion():
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
    iterations = 2      # trajectories to be generated
    prerecorded = False
    dnn_prerecorded = True

    if(not dnn_prerecorded):
        #clfs = train_learner("svm") # not for 3 dim Y
        #clfs = train_learner("lin")
        #clfs = train_learner("lin_bayesian_ridge") # not for 3 dim Y
        #clfs = train_learner("lin_lasso")
        print "training"
        print sklearn.__version__
        #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_100_10.p")
        clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_200_10.p")
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
        output = 1        
        # 0 - cartisean pos
        # 1 - motor pos
        training_set_list = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_3000_10.p",output)
        regressor_list = train_dnn(training_set_list) 
       
        ''' To test and evaluate dnn without executing trajectory
        test_set_list = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_88_20_20.p",output)
        prediction_set_list = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_88_20_20.p",output)
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
    
    if (dnn_prerecorded):
        #q_dict = app.rave.get_config_dict()    
        t = app.rave.get_manip_frame("right_arm")
        data = []
        f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_100_10.p","rb")
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
  
        for i in range(iterations):
            app.rave.clear_graphic_handles("*")
            print "dnn prerecorded iteration:",i+1
            #traj = app.calculate_circle_positions(num_points, radius):
            start = (i*num_points)+i
            end = (i*num_points+num_points)+i+1
            trial = data[start:end]   
            print len(trial), start, end 
            app.evaluate_data_from_file_dnn(regressor_list, trial, i)
            # multiple learners
            #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
            sleep(1)
        return    
    if (prerecorded):
        #q_dict = app.rave.get_config_dict()    
        t = app.rave.get_manip_frame("right_arm")
        data = []
        f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_100_10.p","rb")
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