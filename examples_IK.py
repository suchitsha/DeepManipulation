# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (420, 720, -1, -1),
#  'transitions': []}
### end of header
from pyutils.matrix import *

def init(self):
    pass

def execute(self):
    idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([-1.2, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0.    ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)
    sleep(1)

    q = app.rave.get_robot_config()
    q_dict = app.rave.get_config_dict()
    print q_dict["right_arm"]
    
    tr = app.rave.get_manip_frame("right_arm")
    app.rave.add_coord("tr", tr)
    print "tr"
    print tr
    print tr[0:3,3]
    
    offset = txyz(0,0,0.1)
    goal = dot(offset, tr)
    app.rave.add_coord("goal", goal)
    print goal
    q_goal = app.rave.find_ik("right_arm", goal)
    
    idle_cfg["right_arm"] = q_goal
    app.rave.set_config_dict(idle_cfg)
    print q_goal
    
