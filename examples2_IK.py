# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (530, 720, -1, -1),
#  'transitions': []}
### end of header
from pyutils.matrix import *

def test_reachability(goal):
    app.rmap.load_reachability_map("right_arm") 
    map_frame = app.rmap.update_map_position()
    target_frame = dot(inv(map_frame), goal)
    
    r_index = app.rmap.check_reachability(target_frame[0:4,3])
    return r_index  
            

def find_ik_numerically(goal, i):
    #only right arm
    q_goal = app.rave.find_ik("right_arm", goal)
    if not len(q_goal):
        print "unable to find ik solution for offset at step %s" %i
        return None
    
    offset_cfg = app.rave.get_config_dict()
    offset_cfg["right_arm"] = q_goal
    return offset_cfg

def find_ik_analytically(goal, i):
    #includes torso motion
    offset_cfg = app.rave.get_config_dict()
    try:
        q_goal = app.ikine.dual_ik(None, goal, offset_cfg)
    except:
        print "unable to find ik solution for offset at step %s" %i
        return None   
    offset_cfg.update(q_goal)
    return offset_cfg

def init(self):
    pass

def execute(self):
    app.rave.clear_graphic_handles("*")
    print "execute of %s called!" % self.name
    for op, data in app.wipe.play_list:
        if op != "tree":
            continue
        for i, config in enumerate(data):
            app.rave.set_robot_config(config)
            sleep(0.1)           
            tr = app.rave.get_manip_frame("right_arm")
            app.rave.add_coord("tr", tr)
                       
            # rational: in an innner loop, have the circle offset
            offset = txyz(0,0.25,0.0)
            goal = dot(offset, tr)
            app.rave.add_coord("goal", goal)
            
            
            app.rave.clear_graphic_handles()
    
            print test_reachability(goal)
            
            #offset_cfg = find_ik_numerically(goal, i)
            offset_cfg = find_ik_analytically(goal, i)
            
            if offset_cfg is None:
                continue
            app.rave.set_config_dict(offset_cfg)
            sleep(0.001) 
                 