# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (460, 200, -1, -1),
#  'transitions': []}
### end of header
def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    idle_cfg = {
        'head': array([-0.4, 0.6]), 
        'right_arm': array([-0.9, -1.5, -0.2909,  1.8,  0.    ,  -0.0    , -0.0]),
        'right_hand': array([-0.1, 0.2, 0.157, 0, 0.4, 0.2, 0, 0.4, 0.2, 0, 0.4, 0.2]),  
        'torso': array([ -0.0, -0.8, 1.6 ]), 
        'left_hand': array([0, 0.6, 0.6]*4), 
        'left_arm': array([-1.0, -1.5, -0.2909,  1.1,  0. ,  -0.0    , -0.0])
    }
    app.rave.set_config_dict(idle_cfg)