# {'exits': ['out'],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (960, 240, -1, -1),
#  'transitions': [('out', 'circle_dnn_poly_multipl_lrnrs_simple2')]}
### end of header
def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    return dict(exit="out")