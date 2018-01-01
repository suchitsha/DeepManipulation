# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (210, 540, -1, -1),
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
# autoencoder adapted from #https://medium.com/towards-data-science/autoencoders-introduction-and-implementation-3f40483b0a85


def init(self):
    app.points_on_trajectory = 20
    app.FEATURES = [""]#, "x", "y", "z"]
    app.COLUMNS = ["",""]# "out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9","out_10","out_11","out_12","out_13","out_14","out_15","out_16","out_17","out_18","out_19","out_20"]        
    app.LABEL = [""]#out_0","out_1","out_2","out_3","out_4","out_5","out_6","out_7","out_8","out_9","out_10","out_11","out_12","out_13","out_14","out_15","out_16","out_17","out_18","out_19","out_20"]
    app.num_images = 10000
    app.dir = []
    app.lr = 0.001
    app.iter = 3000
    app.batch_size = 1000
    pass

def init_dnn():     
    print "Initializing Regressor:"
    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in app.FEATURES]
    #print "feature", feature_cols
    dim = app.points_on_trajectory
    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[2048,1024, 512, 256, 256, 256, 256, 256, 256],
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
    ''' 
    point1 = [randint(0, 100),randint(0, 100)] 
    point2 = point1
    jumph = 2
    jumpv = 7
    lx = []
    ly = []
    direction_codes = []
    for i in range(app.points_on_trajectory):
        direction = randint(0, 3)
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
    plt.plot(lx,ly,"ks-", lw=8)        
    plt.axis('off')
    #plt.show()
    #plt.figure(figsize=(20,5))
    f = '/home/shar_sc/Documents/DirtDetection/data/trajectory_primitives/'+ name + '.png'
    plt.savefig(f, bbox_inches='tight')
    plt.clf()
    plt.close()
    app.dir.append(direction_codes)
    return direction_codes
    
def execute(self):
    print "execute of %s called!" % self.name
    #mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
    #img = mnist.train.images[2]
    #plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
    learning_rate = app.lr#0.001
    # Input and target placeholders
    inputs_ = tf.placeholder(tf.float32, (None, 128,128,1), name="input")
    targets_ = tf.placeholder(tf.float32, (None, 128,128,1), name="target")

    ### Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 128x128x16
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 64x64x16
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 64x64x8
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 32x32x8
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 32x32x8

    conv3_1 = tf.layers.conv2d(inputs=encoded, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    encoded2 = tf.layers.max_pooling2d(conv3_1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 16x16x8

    ### Decoder        
    upsample1 = tf.image.resize_images(encoded2, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 32x32x8
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 32x32x8
    upsample2 = tf.image.resize_images(conv4, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 64x64x8
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 64x64x8
    upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 128x128x8
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
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

    #init regressor
    regressor = init_dnn()

    ''' To generate new images
    for i in range(app.num_images):
        dirc = generate_images('fig_'+str(i))
        #print("directions:",i , dirc)
    
    #save values
    data_file = open("/home/shar_sc/Documents/DirtDetection/data/trajectory_primitives/direc.p","wb")#"a+b")#wb
    for data in app.dir:    
        pickle.dump(data, data_file)
    data_file.close()
    #'''
    images_in_batch = []    
    images_in = []
    # for testing
    images_test = []
    #read images
    for i in range(app.num_images):
        img = cv2.imread('/home/shar_sc/Documents/DirtDetection/data/trajectory_primitives/fig_' + str(i) + '.png',0)
        img = cv2.resize(img,dsize=(128,128) , interpolation = cv2.INTER_CUBIC)
        #cv2.imshow('image',img)
        #Numpy array
        np_image_data = np.asarray(img)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data,axis=0)
        np_final = np_final.reshape((128,128,-1))   
        
        if i < 10:
            images_test.append(np_final)
        else:
            images_in.append(np_final)   
            #'''1
            if i%app.batch_size==0 :
                images_in_batch.append(images_in)
                images_in = []
                
            #'''
    
    print ("Training size/", app.batch_size, ":", len(images_in_batch))
    print ("Test size:", len(images_test))    
    sess = tf.Session()
    
    epochs = app.iter
    
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        #for ii in range(mnist.train.num_examples//batch_size):
            #batch = mnist.train.next_batch(batch_size)
            #imgs = batch[0].reshape((-1, 28, 28, 1))
        for j in range(len(images_in_batch)):   
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: images_in_batch[j], targets_: images_in_batch[j]})
            #batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: images_in, targets_: images_in})
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(batch_cost))

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
    in_imgs = images_test
    reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs})# .reshape((10, 28, 28, 1))})

    for images, row in zip([in_imgs, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((128, 128)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    #'''
    #To read data
    objs = []
    f = open("/home/shar_sc/Documents/DirtDetection/data/trajectory_primitives/direc.p","rb")
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    #print len(objs)
    #print objs
    
    #for testing
    val_in = None
    
    for j in range(len(images_in_batch)):
        images_in1 = images_in_batch[j]
        for i in range(len(images_in1)):
            #get value of encoded image
            in_imgs1 = []    
            in_imgs1.append(images_in1[i])
            encoded_img = sess.run(encoded2, feed_dict={inputs_:in_imgs1})#.reshape((1, 128, 128, 1))})
            #print( encoded_img )
            inp = np.asarray(objs[j*app.batch_size + i])
            inp = inp.astype(np.float32)#np.int32
            #print inp

            def get_train_inputs():
                x = tf.constant(encoded_img)  # ([ [  e_x, e_y, e_z ] ])             
                y = tf.reshape(tf.constant([   [inp[0]], [inp[1]],[inp[2]],[ inp[3] ],[inp[4] ],[inp[5]],[inp[6]],[inp[7]],[ inp[8] ],[ inp[9] ],[ inp[10] ],[inp[11] ],[inp[12]],[inp[13]],[inp[14]],[ inp[15] ],[ inp[16] ],[ inp[17] ],[inp[18] ],[inp[19]]   ]), [1,20])
                return x, y

            #inp_y = [[   [inp[0]], [inp[1]],[inp[2]],[ inp[3] ],[inp[4] ],[inp[5]],[inp[6]],[inp[7]],[ inp[8] ],[ inp[9] ],[ inp[10] ],[inp[11] ],[inp[12]],[inp[13]],[inp[14]],[ inp[15] ],[ inp[16] ],[ inp[17] ],[inp[18] ],[inp[19]]  ] ]
            #regressor.partial_fit(x=encoded_img,y=inp_y, steps=1)
            print("fit image:" , i,j)
            regressor.partial_fit(input_fn=get_train_inputs,steps=1)#x=encoded_img,y=inp_ten,steps=1) #
            val_in = encoded_img
        '''
        if j == 4:
            break
       '''     
    #predict
    out = regressor.predict(x=val_in)
    # .predict() returns an iterator; convert to a list and print predictions
    predictions = list(itertools.islice(out, 1))
    print("Predictions: {}".format(str(predictions)))
    #'''    
    # close session    
    sess.close()  

    return
