# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 290, -1, -1),
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

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    learnMotion()
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
        #'''
        temp_x = [objs[i][0],objs[i][1][0],objs[i][1][1],objs[i][1][2]]
        el_X.append(temp_x)
        temp_y = [objs[i][2][0], objs[i][2][1], objs[i][2][2]]
        el_y.append(temp_y)
        #'''
        # to train over x= time,pos --> positions y= motor_pos_delta --> motor positions delta
        '''
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

def evaluate(clf, pos_x, pos_y, pos_z, num_points, radius):
    s_coord = cartToSpher(pos_x,pos_y,pos_z)
    sum_d = 0.0
    sum_theta = 0.0
    sum_phi = 0.0
    
    #for error
    sum_x = pos_x
    sum_y = pos_y
    sum_z = pos_z
    total_error = 0.0
    
    idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([0, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0.    ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)
    app.rave.remove_coords("*")
    app.rave.clear_graphic_handles("*")
    q_dict = app.rave.get_config_dict()    
    #tr = app.rave.get_link_frames(["rightarm1"])["rightarm1"]
    tr = app.rave.get_manip_frame("right_arm")
    #app.rave.add_coord("tr", tr)
    offset = txyz(pos_x,pos_y,pos_z)
    center_frame = dot(offset, tr)
    center_frame[0:3,0:3] = dot(roty(pi/2.), rotz(pi/2))[0:3,0:3]
    app.rave.add_coord("goal", center_frame,"small")
    #print q_dict["right_arm"] 
    #print "center_frame:",center_frame
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
    
    for i in range(num_points+1):
        result = clf.predict([[i, s_coord[0] , s_coord[1], s_coord[2] ]])
        
        #'''
        # For cartesian positions learned
        #print 'result:' , result
        sum_d += result[0][0]
        sum_theta += result[0][1]
        sum_phi += result[0][2]
        goal_s_x = s_coord[0] + sum_d
        goal_s_y = s_coord[1] + sum_theta
        goal_s_z = s_coord[2] + sum_phi          
        goal_cart = spherToCart(goal_s_x, goal_s_y, goal_s_z)
        #print "goal_cart:",goal_cart
        #result_cart = cartToSpher(result[0][0],result[0][1],result[0][2])
        sum_cart = spherToCart(sum_d, sum_theta, sum_phi)
        #print 'goal:', goal_cart 
        
        '''
        #execute trajectory
        tr1 = app.rave.get_manip_frame("right_arm")
        #app.rave.add_coord("tr", tr)
        offset1 = txyz(goal_cart[0],goal_cart[0],goal_cart[0])
        goal_frame = dot(offset1, tr1)
        app.rave.add_coord("destination", goal_frame)
        app.rave.add_coord("destination1", offset1)
        '''
        offset = center_frame
        offset[0,3] = goal_cart[0] + .65 #-.35 for lissajous # .65 for circle
        offset[1,3] = goal_cart[1] 
        offset[2,3] = goal_cart[2] + .90
        #print "offset:",offset
        app.rave.add_coord("destination", offset,"small")
        
        ##points.append(offset[0:3,3])
        
        #initial = app.rave.get_config_dict()
        q_dict = app.find_ik_rotational(offset, i)
        if q_dict is None:
            #print "no solution"
            #colors.append((1, 0, 0))
            #app.rave.draw_line("line", points, colors, 4)
            continue
        #qd = q_dict["right_arm"] - initial["right_arm"] 
        #if i > 1 and (any(qd > 0.3) or any(qd < -0.3)): #12deg
        #    print "re-configuration not allowed"
        #    return  
          
        colors.append((0, 1, 0))
        #app.rave.draw_line("line", points, colors, 4)
        
        app.rave.set_config_dict(q_dict)   
        #reached_frame = app.rave.get_manip_frame("right_arm")
        #app.rave.add_coord("circle%d" % (i), reached_frame, "small")  
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
        
        expected = dot(txyz(*tr[0:3,3]), txyz(exp_x, exp_y, exp_z))
        
        error = calc_error(reached_frame[0:3,3], exp_x, exp_y, exp_z)
        #app.rave.add_coord("expected%d" % (i), expected, "small")  
        
        print "error circle:", error
        total_error += error
        #time.sleep(0.5)
        
        points_expected.append(expected[0:3,3])
        #print "expected:",expected[0:3,3]
        #app.rave.draw_line("line_expected", points_expected, [0,1,0], 4)
        app.rave.draw_line("line", points, [0,0,1], 4)
    
    print "total error circle:", total_error/num_points
    return

def calc_error(g_cart, x,y,z):
    error = math.sqrt( math.pow((g_cart[0] - x),2) + math.pow((g_cart[1] - y),2) + math.pow((g_cart[2] - z),2) )
    return error  

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
    clf = MultiOutputRegressor(GradientBoostingRegressor(loss='lad'), n_jobs=-1).fit(X, y)
    #clf = MultiOutputRegressor(RandomForestRegressor(max_depth=30, random_state=0)).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor() )).fit(X, y)
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
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_88_1000_20.p")
    clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_circle60.p")#data2_8888_1750_60.p")
    
    #clfs = train_learner("multi","/home/shar_sc/Documents/DirtDetection/data/motion/data2_12900_10.p")
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
    iterations = 1      # trajectories to be generated
    prerecorded = False
    print "evaluating"
    if (prerecorded):
        for i in range(iterations):
            print "prerecorded iteration:",i+1
            data = []
            f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_200_10.p","rb")
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break        
            evaluate(clfs[0], data[i][1][0], data[i][1][1], data[i][1][2], num_points, radius)
            # multiple learners
            #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
            time.sleep(3)  
        return
    for i in range(iterations):
        print "iteration:",i+1
        pos_x = random.random()*(max_x - margin - min_x - margin) + (min_x + margin)
        pos_y = random.random()*(max_y - margin - min_y - margin) + (min_y + margin)
        pos_z = random.random()*(max_z - margin - min_z - margin) + (min_z + margin)
        #pos_x = 1.0 pos_y = -0.1 pos_z = 1.0

        evaluate(clfs[0], pos_x, pos_y, pos_z, num_points, radius)
        # multiple learners
        #eval_multiple(clfs, pos_x, pos_y, pos_z, num_points, radius)        
        time.sleep(3)    
    return
