# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (220, 290, -1, -1),
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

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct,ConstantKernel)
from sklearn.semi_supervised import LabelSpreading
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from time import time
import logging
                                              
def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    #print(__doc__)
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
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
    #print objs
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
        client.send_goal_and_wait(goal, execute_timeout=rospy.Duratione(order_timeout) )
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
    
    #clf = MultiOutputRegressor(KernelRidge(alpha=1, kernel='rbf', gamma=None, degree=7, coef0=1, kernel_params=None)).fit(X, y)
    #clf = MultiOutputRegressor( SGDRegressor(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=50, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False) ).fit(X, y)
    #clf = MultiOutputRegressor(  neighbors.KNeighborsRegressor(30, weights='distance', algorithm=‘brute’,leaf_size=100   ) ).fit(X, y)

    #good one
    #clf = MultiOutputRegressor(  neighbors.KNeighborsRegressor(10, weights='uniform'   ) ).fit(X, y)
    '''
    kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0)),
           ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)]
    clf = MultiOutputRegressor( GaussianProcessRegressor(kernel=kernels[4], alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)  ).fit(X, y)
    #'''
    #clf = MultiOutputRegressor( MLPRegressor(hidden_layer_sizes=(200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200 ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08) ).fit(X, y) 
    
    #best
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

def learnMotion():
    
    #clfs = train_learner("svm") # not for 3 dim Y
    #clfs = train_learner("lin")
    #clfs = train_learner("lin_bayesian_ridge") # not for 3 dim Y
    #clfs = train_learner("lin_lasso")
    print "training"
    print sklearn.__version__
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data2_40_10.p")
    clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_circle60.p") #data2_40_10.p")
    
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data2_12900_10.p")
    
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_8_20_20.p")
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_88_20_20.p")
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_88_1000_20.p")
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data2_8888_2950_60.p")
    #clfs = train_learner("rf")
    
    #two learners at a time
    #clfs.extend(train_learner("multi","data5.p"))
    
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
    prerecorded = True
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
    
    if (prerecorded):
        #q_dict = app.rave.get_config_dict()    
        t = app.rave.get_manip_frame("right_arm")
        data = []
        f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_200_10.p","rb")
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
            #print len(trial), start, end 
            app.evaluate_data_from_file(clfs[0], trial, i)
            # multiple learners
            #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
            sleep(2)
        
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
        time.sleep(2)    
    return
