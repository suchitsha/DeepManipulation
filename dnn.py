# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (200, 720, -1, -1),
#  'transitions': []}
### end of header
import itertools

import pandas as pd
import tensorflow as tf
import cPickle as pickle

from random import randint

COLUMNS = ["time", "x", "y", "z","out_x","out_y","out_z"]#, "out_y", "out_z"]
FEATURES = ["time", "x", "y", "z"]
LABEL = ["out_x","out_y","out_z"]

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    run_DNN()
    return

def process_data(file_name):
    #data_pos = []
    data_final_pos_x = []
    data_final_pos_y = []
    data_final_pos_z = []    
    #data_motor_pos = []
    list_x = []
    list_y = []
    list_z = []  
    time = []
    f = open(file_name,"rb")
    while 1:
        try:
            line = pickle.load(f)
            time.append(line[0])
            list_x.append(line[1][0])
            list_y.append(line[1][1])
            list_z.append(line[1][2])                        
            #data_pos.append(line[1])
            data_final_pos_x.append(line[2][0])
            data_final_pos_y.append(line[2][1])
            data_final_pos_z.append(line[2][2])                        
            #data_motor_pos.append(line[5])
        except EOFError:
            break
    f.close()
    d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out_x': data_final_pos_x,
         'out_y': data_final_pos_y,
         'out_z': data_final_pos_z
         } 
    df = pd.DataFrame(d)
    return df   
     
def process_data2(file_name):
    #data_pos = []
    data_final_pos_x = []
    data_final_pos_y = []
    data_final_pos_z = []    
    #data_motor_pos = []
    list_x = []
    list_y = []
    list_z = []  
    time = []
    f = open(file_name,"rb")
    count = 0
    r = randint(0,300)
    required = 40
    while 1:
        try:
            line = pickle.load(f)
            count += 1
            if count < r:
                continue
            time.append(line[0])
            list_x.append(line[1][0])
            list_y.append(line[1][1])
            list_z.append(line[1][2])                        
            #data_pos.append(line[1])
            data_final_pos_x.append(line[2][0])
            data_final_pos_y.append(line[2][1])
            data_final_pos_z.append(line[2][2])                        
            #data_motor_pos.append(line[5])
            if count > (r+required):
                break
        except EOFError:
            break
    f.close()
    d = {'time' : time,
         'x' : list_x,
         'y' : list_y,
         'z' : list_z,
         'out_x': data_final_pos_x,
         'out_y': data_final_pos_y,
         'out_z': data_final_pos_z
         } 
    df = pd.DataFrame(d)
    return df   
    
        
def run_DNN():
    tf.logging.set_verbosity(tf.logging.INFO)        
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

    print "feature", feature_cols
    training_set = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_3000_10.p")
    test_set = process_data("/home/shar_sc/Documents/DirtDetection/data/motion/data_100_10.p")
    prediction_set = process_data2("/home/shar_sc/Documents/DirtDetection/data/motion/data_100_10.p")
    
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[1024, 512, 256,256, 128, 64],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=3,
                                            #optimizer=tf.train.ProximalAdagradOptimizer(
                                            #learning_rate=0.1,
                                            #l1_regularization_strength=0.001)
                                            #model_dir="/tmp/learned_model"
                                            )
    # Fit
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=1000)#5000)

    # Score accuracy
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    # Print out predictions
    y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    # .predict() returns an iterator; convert to a list and print predictions
    predictions = list(itertools.islice(y, 20))
    print("Predictions: {}".format(str(predictions)))
    print prediction_set
    avg_error_x = 0.0
    avg_error_y = 0.0
    avg_error_z = 0.0        
    errors = []
    for x in range(len(predictions)):
        error_x = prediction_set['out_x'][x] - predictions[x][0]
        error_y = prediction_set['out_y'][x] - predictions[x][1]
        error_z = prediction_set['out_z'][x] - predictions[x][2]                
        avg_error_x += abs(error_x)
        avg_error_y += abs(error_y)
        avg_error_z += abs(error_z)        
        errors.append([error_x,error_y,error_z])
    avg_error_x = avg_error_x/len(predictions) # divide by 0 exception
    avg_error_y = avg_error_y/len(predictions) # divide by 0 exception
    avg_error_z = avg_error_z/len(predictions) # divide by 0 exception        
    print "errors:",errors
    print "avg:",avg_error_x, avg_error_y, avg_error_z    
    return
                   
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels