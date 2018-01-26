import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
from random import randint
import cPickle as pickle
import cv2
import itertools
import math
#import scipy.misc
#plt.style.use('GTKAgg')
# autoencoder adapted from #https://medium.com/towards-data-science/autoencoders-introduction-and-implementation-3f40483b0a85


#def init(self):
points_on_trajectory = 20
FEATURES = [""]#, "x", "y", "z"]
COLUMNS = ["",""]# "out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9","out_10","out_11","out_12","out_13","out_14","out_15","out_16","out_17","out_18","out_19","out_20"]        
LABEL = [""]#out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9","out_10","out_11","out_12","out_13","out_14","out_15","out_16","out_17","out_18","out_19","out_20"]
num_images = 100#20000#200000
dir1 = []
lr = 0.001
iter1 = 1#0
batch_size = 100
num_of_results = 10
out_dir = '/home_local/shar_sc/auto_result/'
#    pass

def init_dnn():     
    print "Initializing Regressor:"
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
    #print "feature", feature_cols
    dim = points_on_trajectory
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[2048,1024, 512, 256, 256, 256, 256, 256, 256,256, 256, 256, 256, 256, 256,256, 256, 256, 256],
                                            activation_fn=tf.nn.relu,
                                            #dropout=0.1,
                                            #gradient_clip_norm=None,
                                            #enable_centered_bias=True,
                                            label_dimension=dim,
                                            #optimizer=tf.train.ProximalAdagradOptimizer(
                                            #learning_rate=0.001,
                                            #l1_regularization_strength=0.001),
                                            model_dir="/home_local/shar_sc/learn_shapes_from_primitives"
                                            )
    return regressor

def generate_images(name):
    '''
    0 left to right -
    1 down to up |
    2 left to right diagonal upwards /
    3 left to right diagonal down \
    4 up to down |
    5 right to left diagonal down /    
    ''' 
    point1 = [randint(0, 100),randint(0, 100)] 
    point2 = point1
    jumph = 3
    jumpv = 6
    lx = []
    ly = []
    direction_codes = []
    for i in range(points_on_trajectory):
        direction = randint(0, 5)
        lx.append(point1[0])
        ly.append(point1[1])        
        if (direction == 0):
            point2[0] = point2[0] + jumph
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
            point2[0] = point2[0] + jumph 
            point2[1] = point2[1] + jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
            direction_codes.append(direction)
        elif (direction == 3):
            point2[0] = point2[0] + jumph
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
            point2[0] = point2[0] - jumph
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
    f = '/home_local/shar_sc/trajectory_primitives/'+ name + '.png'
    plt.savefig(f, bbox_inches='tight')
    plt.clf()
    plt.close()
    dir1.append(direction_codes)
    return direction_codes


def generate_result(name, int_pred):
    '''
    0 left to right -
    1 down to up |
    2 left to right diagonal upwards /
    3 left to right diagonal down \
    4 up to down |
    5 right to left diagonal down /
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
            point2[0] = point2[0] + jumph
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])        
        elif (direction == 1):
            point2[1] = point2[1] + jumpv
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])        
        elif (direction == 2):
            point2[0] = point2[0] + jumph 
            point2[1] = point2[1] + jumpv 
            point1 = point2
            lx.append(point2[0])
            ly.append(point2[1])                    
        elif (direction == 3):
            point2[0] = point2[0] + jumph
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
            point2[0] = point2[0] - jumph
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

def prediction_to_directions(predictions,p_num):
    mx = float(max(predictions[0]))
    mn = float(min(predictions[0]))
    print "max:", mx
    print "min:", mn
    range_pred = mx - mn
    int_pred = []
    for p in range(points_on_trajectory):
        #interpolate predictions in the range of directions used
        #TODO not perfect it still prints directions +1
        interpolated_prediction =  ( 6.0/range_pred ) * predictions[0][p]  - mn*(6.0/range_pred)
        int_pred.append( math.floor(interpolated_prediction) )
    print "interpolated predictions:", int_pred        
    generate_result(p_num, int_pred)
    return
    
def execute():
    #mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
    #img = mnist.train.images[2]
    #plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
    learning_rate = lr#0.001
    # Input and target placeholders
    inputs_ = tf.placeholder(tf.float32, (None, 128,128,1), name="input")
    targets_ = tf.placeholder(tf.float32, (None, 128,128,1), name="target")
    #TODO increase num of filters
    ### Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 128x128x16
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 64x64x16
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 64x64x8
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 32x32x8
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 32x32x8

    conv3_1 = tf.layers.conv2d(inputs=encoded, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    encoded2 = tf.layers.max_pooling2d(conv3_1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 16x16x8

    ### Decoder        
    upsample1 = tf.image.resize_images(encoded2, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 32x32x8
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    upsample2 = tf.image.resize_images(conv4, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 64x64x8
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 64x64x8
    upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 128x128x8
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 128x128x16

    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
    #Now 128x128x1

    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)

    # Pass logits through sigmoid and calculate the cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

    # Get cost and define the optimizer
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    ''' To generate new images
    for i in range(num_images):
        dirc = generate_images('fig_'+str(i))
        #print("directions:",i , dirc)
    
    #save values
    data_file = open("/home_local/shar_sc/trajectory_primitives/direc.p","wb")#"a+b")#wb
    for data in dir1:    
        pickle.dump(data, data_file)
    data_file.close()
    #'''
    images_in_batch = []    
    images_in = []
    # for testing
    images_test = []
    #read images
    for i in range(num_images):
        img = cv2.imread('/home_local/shar_sc/trajectory_primitives/fig_' + str(i) + '.png',0)
        img = cv2.resize(img,dsize=(128,128) , interpolation = cv2.INTER_CUBIC)
        #cv2.imshow('image',img)
        #Numpy array
        np_image_data = np.asarray(img)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data,axis=0)
        np_final = np_final.reshape((128,128,-1))   
        
        images_in.append(np_final)   
        if (i+1)%batch_size==0 :
            images_in_batch.append(images_in)
            images_in = []
            #print i
            
    print ("Training size/", batch_size, ":", len(images_in_batch), ":" , len(images_in_batch[-1]))
    images_test = images_in_batch[-1][-10:]
    print ("Test size:", len(images_test))    
    
    sess = tf.Session()
    epochs = iter1
    sess.run(tf.global_variables_initializer())
    step_count = 0
    for e in range(epochs):
        # Restore variables from disk.
        #saver.restore(sess, "/home_local/shar_sc/auto_regressor_model/auto_model")
        #print("Model restored.")
        print "len - images_in_batch",len(images_in_batch)    
        for j in range(len(images_in_batch)):   
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: images_in_batch[j], targets_: images_in_batch[j]})
            step_count = step_count + 1
            #batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: images_in, targets_: images_in})
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(batch_cost),
                  "Batch Number: {}".format(j),
                  "Batch size: {}".format( len(images_in_batch[j]) ) )
        save_path = saver.save(sess,"/home_local/shar_sc/auto_regressor_model/auto_model",global_step=step_count) # "/home_local/shar_sc/auto_regressor_model/"
        print("Model saved in file: %s" % save_path)

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
    plt.savefig('/home_local/shar_sc/foo.png')
    in_imgs = images_test
    reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs})# .reshape((10, 28, 28, 1))})

    for images, row in zip([in_imgs, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((128, 128)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig('/home_local/shar_sc/foo1.png')
    plt.clf()
    plt.close()
    
    
    
    
    '''
    #TODO remove this block , prints middle layer
    middle_layer = sess.run(conv3_1, feed_dict={inputs_: in_imgs})# .reshape((10, 28, 28, 1))})
    #10 * 32*32     x =10 * 256*256
    for imi in range(10):
         middle_layer[imi] = middle_layer[imi].reshape(16,16)
         middle_layer[imi] = middle_layer[imi].resize(128,128)
    print middle_layer.shape
    
    for images, row in zip([in_imgs, middle_layer], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((128, 128)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig('/home_local/shar_sc/foo_m.png')
    plt.clf()
    plt.close()
    
    #'''
    
    
    
    
    
    
    
    
    
    #'''
    #To read data
    objs = []
    f = open("/home_local/shar_sc/trajectory_primitives/direc.p","rb")
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    #print len(objs)
    #print objs


    #init regressor
    regressor = init_dnn()
    
    #for testing
    val_in = []
    val_in_img = [] 
    val_in_dir = []   
    for j in range(len(images_in_batch)):
        images_in1 = images_in_batch[j]
        for i in range(len(images_in1)):
            #get value of encoded image
            in_imgs1 = []    
            in_imgs1.append(images_in1[i])
            encoded_img = sess.run(encoded2, feed_dict={inputs_:in_imgs1})#.reshape((1, 128, 128, 1))})
            #print( encoded_img )
            inp = np.asarray(objs[j*batch_size + i])
            inp = inp.astype(np.float32)#np.int32
            #print j*batch_size + i
            print "size",len(encoded_img)
            def get_train_inputs():
                x = tf.constant(encoded_img)  # ([ [  e_x, e_y, e_z ] ])             
                y = tf.reshape(tf.constant([   [inp[0]], [inp[1]],[inp[2]],[ inp[3] ],[inp[4] ],[inp[5]],[inp[6]],[inp[7]],[ inp[8] ],[ inp[9] ],[ inp[10] ],[inp[11] ],[inp[12]],[inp[13]],[inp[14]],[ inp[15] ],[ inp[16] ],[ inp[17] ],[inp[18] ],[inp[19]]   ]), [1,20])
                return x, y

            #inp_y = [[   [inp[0]], [inp[1]],[inp[2]],[ inp[3] ],[inp[4] ],[inp[5]],[inp[6]],[inp[7]],[ inp[8] ],[ inp[9] ],[ inp[10] ],[inp[11] ],[inp[12]],[inp[13]],[inp[14]],[ inp[15] ],[ inp[16] ],[ inp[17] ],[inp[18] ],[inp[19]]  ] ]
            #regressor.partial_fit(x=encoded_img,y=inp_y, steps=1)
            print("fit image:" , i,j)
            regressor.partial_fit(input_fn=get_train_inputs,steps=1)#x=encoded_img,y=inp_ten,steps=1) #
            if ( len(val_in) < num_of_results ):
                val_in.append(encoded_img)
                val_in_img.append(images_in1[i])
                val_in_dir.append(objs[j*batch_size + i])
            #'''#TODO
            if i == 5:
                break
            #'''     
        #'''
        if j == 0:
            break
        #'''     
    
    for p_num in range(len(val_in)):
        #predict
        out = regressor.predict(x=val_in[p_num])
        # .predict() returns an iterator; convert to a list and print predictions
        predictions = list(itertools.islice(out, 1))
        print("Predictions: {}".format(str(predictions)))
        #draw
        prediction_to_directions(predictions,p_num)
    
    for im in range(len(val_in_dir)):
            f = str(im) + '_r'
            generate_result(f, val_in_dir[im])

    # close session    
    sess.close()  
    return

execute()
