# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 200, -1, -1),
#  'transitions': []}
### end of header
import time
import math
import random
import cPickle as pickle
import numpy as np

from sklearn import linear_model
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    return
    
joint_positions = None
joint_velocities = None
joint_accelerations = None
joint_effort = None

def generate_data(fname, num_data_points, max_x, min_x, max_y, min_y, max_z, min_z, margin):

    # create SimpleActionClient object
    client = actionlib.SimpleActionClient('MaterialOrder', common.msg.MaterialOrderAction)
    # wait for action server to start up
    client.wait_for_server()

    data_file = open(fname,"a+b")#"wb")
    for i in range(num_data_points):
        print "iteration:",i+1
        pos_x = random.random()*(max_x - min_x - 2*margin) + (min_x + margin)
        pos_y = random.random()*(max_y - min_y - 2*margin) + (min_y + margin)
        pos_z = random.random()*(max_z - min_z - 2*margin) + (min_z + margin)
        or_x = 0.0
        or_y = 0.0
        or_z = 0.0
        or_w = 0.0

        goal_pos = cartToSpher(pos_x,pos_y,pos_z) 

        goal = common.msg.MaterialOrderGoal()
        goal.order_id = i
        goal.robot_id = 1
        goal.target_pos_x = pos_x
        goal.target_pos_y = pos_y
        goal.target_pos_z = pos_z
        goal.target_orient_x = or_x
        goal.target_orient_y = or_y
        goal.target_orient_z = or_z
        goal.target_orient_w = or_w        

        client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(order_timeout) )
        #print "generate data:", client.get_result()
        time.sleep(3)
        
        #prepare data
        data = [goal_pos[0], goal_pos[1], goal_pos[2], or_x, or_y, or_z, or_w, joint_positions[0],joint_positions[1],joint_positions[2],joint_positions[3],joint_positions[4],joint_positions[5] ]
        print "dumping data:", data
        #write data
        pickle.dump(data, data_file)

    data_file.close()        
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

def callback(data):
    global joint_positions
    global joint_velocities
    global joint_accelerations
    global joint_effort
    #print "Motor Positions:",data.trajectory[-1].joint_trajectory.points[-1].positions # velocities accelerations
    joint_positions = data.trajectory[-1].joint_trajectory.points[-1].positions
    joint_velocities = data.trajectory[-1].joint_trajectory.points[-1].velocities
    joint_accelerations = data.trajectory[-1].joint_trajectory.points[-1].accelerations
    joint_effort = data.trajectory[-1].joint_trajectory.points[-1].effort

                       
def eval(clfs, pos_x, pos_y, pos_z, or_x, or_y, or_z, or_w):
    topic = 'visualization_marker_array'
    publisher = rospy.Publisher(topic, MarkerArray, queue_size=10)

    s_coord = cartToSpher(pos_x,pos_y,pos_z)
    result = []
    for i in range(len(clfs)):
        result.extend( clfs[i].predict([[s_coord[0] , s_coord[1], s_coord[2],or_x, or_y, or_z, or_w ]]) )
        #print "aaaaaa",result[0][0]
    # create SimpleActionClient object
    client = actionlib.SimpleActionClient('MaterialOrder', common.msg.MaterialOrderAction)
    # wait for action server to start up
    client.wait_for_server()
    
    goal = common.msg.MaterialOrderGoal()
    goal.order_id = int(random.random()*1000)
    goal.robot_id = 1
    goal.target_pos_x = pos_x
    goal.target_pos_y = pos_y
    goal.target_pos_z = pos_z
    goal.target_orient_x = or_x
    goal.target_orient_y = or_y
    goal.target_orient_z = or_z
    goal.target_orient_w = or_w
    
    # execute motion
    client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(order_timeout) )
    #print "motion controller:", client.get_result()
    
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = .0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = pos_x
    marker.pose.position.y = pos_y
    marker.pose.position.z = pos_z
    marker.id = 1
    markerArray = MarkerArray()
    markerArray.markers.append(marker)
                
    time.sleep(3)
    
    #move to predicted position
    
    robot_id = rospy.get_param("robot/robot_id")
    robot_timeout = rospy.get_param("robot/robot_timeout")
    feedback_rate  = rospy.get_param("robot/feedback_rate")
    position_tolerance = rospy.get_param("motion_planning/position_tolerance")
    orient_tolerance = rospy.get_param("motion_planning/orient_tolerance")
    planning_engine = rospy.get_param("motion_planning/planning_engine")
    
    robot = moveit_commander.RobotCommander()
    arm_group = moveit_commander.MoveGroupCommander("robot_arm_group")
    arm_group.set_planning_time(robot_timeout)
    arm_group.set_planner_id(planning_engine)
    arm_group.set_goal_position_tolerance(position_tolerance)
    arm_group.set_goal_orientation_tolerance(orient_tolerance)
    arm_group.allow_replanning(True)
    
    fk_service_client = rospy.ServiceProxy('/compute_fk', moveit_msgs.srv.GetPositionFK)
    header = Header()
    header.frame_id =  "base_link"
    state = robot.get_current_state()
    
    
    # put expected positions
    #print "state before change", state    
    state_pos = list(state.joint_state.position)
    for p in range(len(state_pos)): 
        state_pos[p] = result[0][p]
    state.joint_state.position = state_pos #tuple(state_pos)
    #print "state after change" , state    
    #print "11",header     #print "22", robot.get_link_names() 
    
    positions_old = fk_service_client(header, robot.get_link_names(), robot.get_current_state()) # old state
    positions = fk_service_client(header, robot.get_link_names(), state) # new state
    #print "-------------------------------------"
    print "Old Positions from FK" , positions_old.pose_stamped[8].pose.position.x, positions_old.pose_stamped[8].pose.position.y, positions_old.pose_stamped[8].pose.position.z 
    print "Expected Positions from FK" , positions.pose_stamped[8].pose.position.x, positions.pose_stamped[8].pose.position.y, positions.pose_stamped[8].pose.position.z     
    print "Actual positions" , pos_x,pos_y,pos_z   


    marker1 = Marker()
    marker1.header.frame_id = "base_link"
    marker1.type = marker.SPHERE
    marker1.action = marker.ADD
    marker1.scale.x = 0.2
    marker1.scale.y = 0.2
    marker1.scale.z = 0.2
    marker1.color.a = 1.0
    marker1.color.r = .0
    marker1.color.g = .0
    marker1.color.b = 1.0
    marker1.pose.orientation.w = 1.0
    marker1.pose.position.x = positions.pose_stamped[8].pose.position.x
    marker1.pose.position.y = positions.pose_stamped[8].pose.position.y
    marker1.pose.position.z = positions.pose_stamped[8].pose.position.z
    marker1.id = 2
    markerArray.markers.append(marker1)
    publisher.publish(markerArray)

    #get motor positions
    #rospy.Subscriber("/receive_order/result_planned_path", String, callback)
    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()
    #time.sleep(1)
    
    #calculate error
    print "prediction:", result[0]
    print "expected:", joint_positions
    m_error = calc_error_m( result[0] ) #result is array inside array, so [0] 
    print "motor position error percentage:", m_error
    
    p_error = calc_error_pos(pos_x,pos_y,pos_z , positions.pose_stamped[8].pose.position.x, positions.pose_stamped[8].pose.position.y, positions.pose_stamped[8].pose.position.z)
    print "position error", p_error
    print "--------------------------------------"
    return


def calc_error_pos(x,y,z, ex,ey,ez):
    error = math.sqrt( math.pow( (x- ex),2) + math.pow( (y-ey),2) + math.pow( (z-ez),2) )
    return error  
          
def calc_error_m(result):
    error = 0.0
    error2 = 0.0
    #for i in xrange(4):
    for i in xrange(len(result)):
        error += abs( (joint_positions[i] - result[i])/joint_positions[i] )
        error2 += abs(joint_positions[i] - result[i])    
    error /= float(len(result))
    error2 /= float(len(result))
    print "motor position average error:", error2
    error *= 100.
    return error  

def learnIK():
    # workspace for robot
    max_x = 1.5
    min_x = 0.5
    max_y = 1.0
    min_y = -0.9
    max_z = 1.5
    min_z = 0.0
    margin = 0.25
    
    
    #'''# generate data
    fname = "data_IK.p"
    num_data_points = 20000 
    generate_data(fname, num_data_points, max_x, min_x, max_y, min_y, max_z, min_z, margin)
    return "finished data generation"
    #'''
    
    # train
    #clfs = train_IK("svm") # not for 3 dim Y
    #clfs = train_IK("lin")
    #clfs = train_IK("lin_bayesian_ridge") # not for 3 dim Y
    #clfs = train_IK("lin_lasso")
    clfs = train_IK("multi","data_IK.p")
    #clfs = train_IK("rf")
    
    #two learners at a time
    #clfs.extend(train_IK("multi","data5.p"))
    
    # generate random points and evaluate
    iterations = 10     # num of points to be tested
    for i in range(iterations):
        print "iteration:",i+1
        pos_x = random.random()*(max_x - min_x - 2*margin) + (min_x + margin)
        pos_y = random.random()*(max_y - min_y - 2*margin) + (min_y + margin)
        pos_z = random.random()*(max_z - min_z - 2*margin) + (min_z + margin)
        #pos_x = 1.0
        #pos_y = -0.1
        #pos_z = 1.0
        or_x = 0.0
        or_y = 0.0
        or_z = 0.0
        or_w = 0.0
        eval(clfs, pos_x, pos_y, pos_z, or_x, or_y, or_z, or_w)        
        #time.sleep(3)
    
    return "finished evaluation"

def train_IK(name,fname):

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
        temp_x = [objs[i][0], objs[i][1], objs[i][2], objs[i][3], objs[i][4], objs[i][5], objs[i][6]]
        el_X.append(temp_x)
        temp_y = [objs[i][7], objs[i][8], objs[i][9], objs[i][10], objs[i][11], objs[i][12]]
        el_y.append(temp_y)
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

#models
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
    #clf = MultiOutputRegressor(RandomForestRegressor(max_depth=30, random_state=0)).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    #clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=300, random_state=np.random.RandomState(1) )).fit(X, y)
    clf = MultiOutputRegressor( AdaBoostRegressor(DecisionTreeRegressor() )).fit(X, y)
    return clf    

def rf(X,y):
    clf = RandomForestRegressor(max_depth=30, random_state=2)
    clf.fit(X,y)
    return clf    

if __name__ == '__main__':
    try:
        # Initialize node so SimpleActionClient can subscribe over ROS.
        rospy.init_node('learnIK')
        rospy.Subscriber("/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, callback)
        result = learnIK()
        print result
        #rospy.spin()
        #exit()
    except rospy.ROSInterruptException:
        rospy.logerror("LearnIK: Process interupted")    