# {'exits': ['out'],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (1030, 410, -1, -1),
#  'transitions': [('out', 'circle_dnn_poly_multipl_lrnrs_simple')]}
### end of header
def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    return dict(exit="out")