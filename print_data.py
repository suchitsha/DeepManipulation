# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (610, 200, -1, -1),
#  'transitions': []}
### end of header
import cPickle as pickle

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    data = []
    f = open("/home/shar_sc/Documents/DirtDetection/data/motion/data_rec_broomL.p","rb")#accum2.p","rb")#scrub1.p","rb")#accum2.p","rb")#data_200_10.p","rb")
    while True:
        try:
            cur = pickle.load(f)
            data.append(cur)
            print cur
        except EOFError:
            break
