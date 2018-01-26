# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 20, -1, -1),
#  'transitions': []}
### end of header
import cv2
#import tensorflow
#import subprocess

def init(self):
    execute(self)
    #pass

def execute(self):
    print "execute of %s called!" % self.name
    #train, there are some hard coded too
    app.default_random_brightness = 0        
    app.default_random_scale = 0
    app.default_random_crop = 0
    app.default_flip_left_right = False
    app.default_final_tensor_name = 'final_result'


    #''' original
    app.default_bottleneck_dir = '/home/shar_sc/Documents/DirtDetection/tf_files1/bottlenecks'
    app.default_model_dir = '/home/shar_sc/Documents/DirtDetection/tf_files1/inception'
    #'''
    '''#for testing
    app.default_bottleneck_dir = '/home_local/shar_sc/temp/bottlenecks'
    app.default_model_dir = '/home_local/shar_sc/temp/inception'
    '''
    
    
    
    
    app.print_misclassified_test_images = True
    app.validation_batch_size = 100
    app.test_batch_size = -1
    app.default_train_batch_size = 100
    app.default_learning_rate = 0.01
    app.default_how_many_training_steps = 10 #400
    
    '''#for testing
    app.default_summaries_dir = '/home_local/shar_sc/temp/summary'
    app.default_output_labels = '/home_local/shar_sc/temp/retrained_labels.txt'
    app.default_output_graph = '/home_local/shar_sc/temp/retrained_graph.pb'
    app.default_image_dir = '/home/shar_sc/Documents/DirtDetection/data/floor'
    '''
    #''' original
    app.default_summaries_dir = '/home/shar_sc/Documents/DirtDetection/tf_files1/summary'
    app.default_output_labels = '/home/shar_sc/Documents/DirtDetection/tf_files1/retrained_labels.txt'
    app.default_output_graph = '/home/shar_sc/Documents/DirtDetection/tf_files1/retrained_graph.pb'
    app.default_image_dir = '/home/shar_sc/Documents/DirtDetection/data/floor'
    #'''
    
    
    
    
    
    #label
    app.test_image_path="/home/shar_sc/Documents/DirtDetection/data/data_small/dirty_floor/image_001.jpeg"
    app.trained_label_path="/home/shar_sc/Documents/DirtDetection/tf_files1/retrained_labels.txt"
    app.trained_graph_path="/home/shar_sc/Documents/DirtDetection/tf_files1/retrained_graph.pb"
    app.cut_off_probability =0.01
    #process_image
    app.list_of_images = []
    app.base_dir = "/home/shar_sc/Documents/DirtDetection/data/test"
    app.out_dir = "/home/shar_sc/Documents/DirtDetection/data/out/"
    app.segmentsHeight = 2#3
    app.segmentsWidth  = 2#3
    app.split_image = False
    #sliding_window
    app.window_width_fraction=0.20 #0.10#0.25
    app.window_height_fraction=0.20 #0.10#0.25
    app.step_size_width= 20#20 #in pixels 100 200
    app.step_size_height=20#20 #in pixels 100 200    
    app.binary_threshold=0.7 #0.70 #0.40 or 50
    
    #learn_motion
    #for circle_dnn_poly_multipl_lrnrs_simple.py
    app.cost = 0.0 # a big value for beginning
       
    return dict(exit="out")### end of gui script