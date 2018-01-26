# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (320, 540, -1, -1),
#  'transitions': []}
### end of header
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
from random import randint
import cPickle as pickle
import cv2
import itertools
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# cnn architecture based on https://matthewearl.github.io/2016/05/06/cnn-anpr/
#this one is inverted trajectory of cnn1

#def init(self):
points_on_trajectory = 7 #50 # 20
FEATURES = [""]#, "x", "y", "z"]
COLUMNS = ["",""]# "out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9","out_10","out_11","out_12","out_13","out_14","out_15","out_16","out_17","out_18","out_19","out_20"]        
LABEL = [""]#out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9","out_10","out_11","out_12","out_13","out_14","out_15","out_16","out_17","out_18","out_19","out_20"]
num_images = 20000#100000#20000#00#10000
dir1 = []
lr = 0.001
iter1 = 10#0
batch_size = 100#0
dir_no = 5 # num of directions
image_dir = '/home_local/shar_sc/cnn_trajectory_primitives4/'
s_direct = '/home_local/shar_sc/cnn_model4/model.ckpt'
out_dir = '/home_local/shar_sc/cnn_result4/'
#    pass

def generate_result(name, int_pred):
    '''
    0 left to right - straight_90
    1 down to up |    straight_0
    2 left to right diagonal upwards / straight_45
    3 left to right diagonal down \  straight_135
    4 up to down |  straight_180
    #               5 right to left diagonal down /
    ''' 
    point1 = [randint(0, 100),randint(0, 100)] 
    point2 = point1
    jumph = 3
    jumpv = 6
    lx = []
    ly = []
    for i in range(points_on_trajectory):
        direction = int_pred[i] #randint(0, 3)
        lx.append(point1[0])
        ly.append(point1[1])        
        if (direction == 0):
            point2[0] = point2[0] - jumph
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])        
        elif (direction == 1):
            point2[1] = point2[1] + jumpv
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])        
        elif (direction == 2):
            point2[0] = point2[0] - jumph 
            point2[1] = point2[1] + jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
        elif (direction == 3):
            point2[0] = point2[0] - jumph
            point2[1] = point2[1] - jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
        elif (direction == 4):
            point2[0] = point2[0] #+ jumph
            point2[1] = point2[1] - jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
        elif (direction == 5):
            point2[0] = point2[0] + jumph
            point2[1] = point2[1] - jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
    plt.plot(lx,ly,"ks-", lw=8)        
    plt.axis('off')
    #plt.show()
    #plt.ion()
    #plt.figure(figsize=(20,5))
    f = out_dir + str(name) + '.png'
    plt.savefig(f, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def generate_images(name):
    '''
    0 left to right -
    1 down to up |
    2 left to right diagonal upwards /
    3 left to right diagonal down \
    4 up to down |
    #                   5 right to left diagonal down /     
    ''' 
    point1 = [randint(0, 100),randint(0, 100)] 
    point2 = point1
    jumph = 3
    jumpv = 6
    lx = []
    ly = []
    direction_codes = []
    for i in range(points_on_trajectory):
        direction = randint(0, (dir_no -1))
        lx.append(point1[0])
        ly.append(point1[1])        
        if (direction == 0):
            point2[0] = point2[0] - jumph
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])        
            direction_codes.append(direction)
        elif (direction == 1):
            point2[1] = point2[1] + jumpv
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])        
            direction_codes.append(direction)
        elif (direction == 2):
            point2[0] = point2[0] - jumph 
            point2[1] = point2[1] + jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
            direction_codes.append(direction)
        elif (direction == 3):
            point2[0] = point2[0] - jumph
            point2[1] = point2[1] - jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
            direction_codes.append(direction)
        elif (direction == 4):
            point2[0] = point2[0] #+ jumph
            point2[1] = point2[1] - jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])             
            direction_codes.append(direction)                   
        elif (direction == 5):
            point2[0] = point2[0] + jumph
            point2[1] = point2[1] - jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])             
            direction_codes.append(direction)  
                        
    plt.plot(lx,ly,"ks-", lw=8)        
    plt.axis('off')
    #plt.show()
    #plt.ion()
    #plt.figure(figsize=(20,5))
    f = image_dir+ name + '.png'
    #f = '/home_local/shar_sc/Documents/DirtDetection/data/trajectory_primitives/'+ name + '.png'
    plt.savefig(f, bbox_inches='tight')
    plt.clf()
    plt.close()
    dir1.append(direction_codes)
    return direction_codes
    
def execute():
    #mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
    #img = mnist.train.images[2]
    #plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
    learning_rate = lr#0.001
    # Input and target placeholders
    inputs_ = tf.placeholder(tf.float32, (None, 128,128,1), name="input")
    targets_ = tf.placeholder(tf.float32, (None, points_on_trajectory,dir_no), name="target")

    ### Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 128x128x16
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 64x64x16
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=48, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 64x64x8
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 32x32x8
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 16x16x8

    conv3_1 = tf.layers.conv2d(inputs=encoded, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 16x16x8
    encoded2 = tf.layers.max_pooling2d(conv3_1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 8x8x8
        
    # Dense Layer
    encoded2_flat = tf.reshape(encoded2, [-1, 8 * 8 * 8])
    dense = tf.layers.dense(inputs=encoded2_flat, units=512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense, units=512, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu)
    #TODO try dropout
    #dropout = tf.layers.dropout(inputs=dense3, rate=0.1)#, training=mode == learn.ModeKeys.TRAIN)
    # Logits Layer
    #logits = tf.layers.dense(inputs=dropout, units=20)
    uni = points_on_trajectory*dir_no
    logits = tf.layers.dense(inputs=dense3, units=uni)
    out_shape = tf.reshape(logits,[-1,points_on_trajectory,dir_no] )    
    # Pass logits through sigmoid to get reconstructed image
    #decoded = tf.nn.sigmoid(logits)
    loss = None
    train_op = None

    # Pass logits through sigmoid and calculate the cross-entropy loss
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=out_shape)
    
    # Get cost and define the optimizer
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #init regressor
    #regressor = init_dnn()

    direct = image_dir + 'direc.p'
    ''' To generate new images
    for i in range(num_images):
        dirc = generate_images('fig_'+str(i))
        #print("directions:",i , dirc)
    
    #save values
    data_file = open(direct,"wb")#"a+b")#wb
    #data_file = open("/home/shar_sc/Documents/DirtDetection/data/trajectory_primitives/direc.p","wb")#"a+b")#wb
    for data in dir1:    
        pickle.dump(data, data_file)
    data_file.close()
    #'''
    
    #To read data
    objs = []
    f = open(direct,"rb")
    #f = open("/home/shar_sc/Documents/DirtDetection/data/trajectory_primitives/direc.p","rb")
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    f.close()

    sess = tf.Session()
    epochs = iter1
    sess.run(tf.global_variables_initializer())
    
    # for testing
    images_test = []
    objs_test = []
    objs_test_tensor = []
    step = 0
    #read images
    for e  in range(epochs):
        for i in range(num_images):
            img_direct = image_dir + 'fig_' + str(i) + '.png'
            img = cv2.imread(img_direct,0)
            img = cv2.resize(img,dsize=(128,128) , interpolation = cv2.INTER_CUBIC)
            #cv2.imshow('image',img)
            #Numpy array
            np_image_data = np.asarray(img)
            np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            np_final = np.expand_dims(np_image_data,axis=0)
            np_final = np_final.reshape((128,128,-1))   
            images_in = []
            images_in.append(np_final)
            inp = np.asarray(objs[i])
            inp = inp.astype(np.int32)#np.float32)#np.int32    
            #inp = inp.reshape([-1,20])    
            
            #inp = tf.one_hot(indices = inp, depth = 4, on_value = 1.0, off_value = 0.0, axis = -1)    
            inp = inp.reshape(len(inp), 1)
            onehot_encoder = OneHotEncoder(n_values=dir_no, sparse=False)
            inp = onehot_encoder.fit_transform(inp)
            #print inp
            if len(inp) != points_on_trajectory:
                print "size of labels not correct:", inp#len(inp)
                continue
            if len(inp[0]) != dir_no:
                print "size of labels not correct:", inp#len(inp)
                continue                
            inp = inp.reshape([-1,points_on_trajectory,dir_no])    
            #print inp
            if i < 10:
                if (e == 0):
                    images_test.append(np_final)
                    objs_test.append(objs[i])
                    objs_test_tensor.append(inp)
            else:        
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: images_in, targets_: inp})
                print("Epoch: {}/{}/{}".format(i+1,e+1, epochs), "Training loss: {:.4f}".format(batch_cost))
                step = step +1
        
        save_path = saver.save(sess,s_direct,global_step=step)
        print("Model saved in file: %s" % save_path)
            
        '''            
        if i < 10:
            images_test.append(np_final)
            objs_test.append(objs[i])
        else:
            images_in.append(np_final)   
            objs_in.append(objs[i])
            if (i+1.)%batch_size==0. :
                print"i:", i
                images_in_batch.append(images_in)
                images_in = []
                objs_in_batch.append(objs_in)
                objs_in = []
        '''        
    in_imgs = images_test
    reconstructed = sess.run(logits, feed_dict={inputs_: in_imgs})
    reconstructed = np.asarray(reconstructed).reshape([-1,points_on_trajectory,dir_no])
    for res in range(len(reconstructed)):
        lab_arr = []
        for lab in range(len(reconstructed[res])):
            max_index = argmax(reconstructed[res][lab, :])
            lab_arr.append(max_index)
        print "argmax index array:" , lab_arr
        generate_result(str(res), lab_arr)
        print("reconstructed {}: {}".format(str(res),str(reconstructed[res])))
        generate_result(str(res) + str('_o'),objs_test[res] )
                
    #predictions = list(itertools.islice(reconstructed, 1))
    #predictions = np.asarray(predictions).reshape([20,4]) 
    #print("Predictions: {}".format(str(len(predictions))))
    print "labels:", objs_test
    sess.close()  
    return

