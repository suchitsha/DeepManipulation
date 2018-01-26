# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (90, 20, -1, -1),
#  'transitions': []}
### end of header
from os import listdir
from os.path import isfile, join
#split
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from subprocess import call
#label
import tensorflow as tf, sys


#segmentsHeight = 3
#segmentsWidth  = 3

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    #read directory
    app.list_of_images = [join(app.base_dir, fn) for fn in listdir(app.base_dir) if isfile(join(app.base_dir, fn))]
    #process image    
    for i in app.list_of_images:
        app.current_image = i
        print "current image is: %s" %app.current_image
        if app.split_image:
            print "Spliting image"
            split_image()
        else:        
            print "Sliding window over image"
            slide(app.current_image)
        
    print "processed all the images in the directory"    
    #return dict(exit="done")        
    return dict(exit="out")

def split_image():
    print "Reading Image"
    image = cv2.imread(app.current_image)
    #cv2.imshow('image',image)
    #cv2.waitKey(0) throws error at runtime
    
    height,width = image.shape[:2]
    print "Image size: "
    print image.shape[:2]
    heightSeg = height/app.segmentsHeight
    widthSeg  = width/app.segmentsWidth
    
    i = 1
    for x in xrange(1, app.segmentsWidth+1):
        for y in xrange(1,app.segmentsHeight+1):
            #print str(heightSeg*(y-1)) + " " + str(heightSeg*y)
            #print str(widthSeg*(x-1)) + " " + str(widthSeg*x)
            seg_image = image[heightSeg*(y-1):heightSeg*y,widthSeg*(x-1):widthSeg*x]    
            #show and write image segments               
            name = 'segment' + str(i)
            #cv2.imshow(name,segImage)
            #cv2.waitKey(1)
            
            #write file
            cv2.imwrite(app.out_dir + name + '.jpeg',seg_image)

            #label image with classifier
            print "Labels for %s: " % name
            label(name,seg_image)

            i=i+1

    cv2.destroyAllWindows()
    return
    
def label(name, seg_image):
    import tensorflow as tf, sys
    #label image
    image_path = app.out_dir + name + '.jpeg'
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    #image_data = app.window_frame

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile(app.trained_label_path)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(app.trained_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictitfons = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            
        #return single label based on probabiity    
        print '---------------------------------'
        #print label_lines
        #print top_k
        #print predictions[0][top_k[0]]
        #print app.cut_off_probability    
        if predictions[0][top_k[0]] >= app.cut_off_probability:
            if('d' == label_lines[top_k[0]].split()[0][0]):
                print 'Prediction is DIRTY: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]]
            elif('c' == label_lines[top_k[0]].split()[0][0]):     
                print 'Prediction is CLEAN: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]]
            else:
                print 'Class label not parsed correctly'    
        else:
            prob_dirty=0.0
            prob_clean=0.0
            for x in range(5):
                if('d' == label_lines[top_k[x]].split()[0][0]):
                    prob_dirty += predictions[0][top_k[x]]
                elif('c' == label_lines[top_k[x]].split()[0][0]):
                    prob_clean = prob_clean + predictions[0][top_k[x]]
                else:
                    print 'Class label not parsed correctly'
                
            if prob_dirty >= prob_clean:
               print 'Prediction is DIRTY,highest: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]] ,' and ',prob_dirty 
            else:
               print 'Prediction is CLEAN, highest: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]] ,' and ',prob_clean            
        print '---------------------------------'
    return            
    #return dict(exit="all_processed")### end of gui script    




def process_prob(predictions,label_lines):
  
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        #print('%s (score = %.5f)' % (human_string, score))
        
    #return single label based on probabiity    
    #print '---------------------------------'
    #print label_lines
    #print top_k
    #print predictions[0][top_k[0]]
    #print app.cut_off_probability    
    if predictions[0][top_k[0]] >= app.cut_off_probability:
        if('d' == label_lines[top_k[0]].split()[0][0]):
            #print 'Prediction is DIRTY: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]]
            #write on hm prob : predictions[0][top_k[0]]
            return predictions[0][top_k[0]]
        elif('c' == label_lines[top_k[0]].split()[0][0]):     
            #print 'Prediction is CLEAN: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]]
            #write on hm prob clean:
            return 0.0
        else:
            print 'Class label not parsed correctly'    
    else:
        prob_dirty=0.0
        prob_clean=0.0
        num_dirty=0
        num_clean=0
        for x in range(5):
            if('d' == label_lines[top_k[x]].split()[0][0]):
                prob_dirty += predictions[0][top_k[x]]
                num_dirty+=1
            elif('c' == label_lines[top_k[x]].split()[0][0]):
                prob_clean += predictions[0][top_k[x]]
                num_clean+=1
            else:
                print 'Class label not parsed correctly'
            
        if prob_dirty >= prob_clean:
            if num_dirty==0:
                prob_dirty=0.0
            else:
                prob_dirty= prob_dirty/num_dirty
            #write on hm prob :    
            #print 'Prediction is DIRTY,highest: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]] ,' and ',prob_dirty 
            return prob_dirty
        else:
            if num_clean==0:
                prob_clean=0.0
            else:
                prob_clean= prob_clean/num_clean           
            #write on hm prob clean :
            #print 'Prediction is CLEAN,highest: ', label_lines[top_k[0]] ,'with probability ' , predictions[0][top_k[0]] ,' and ',prob_clean            
            return 0.0
                    
        #print '---------------------------------'
    return 0.0          


def slide(image_path):
    image = cv2.imread(image_path)
    i_height,i_width = image.shape[:2]
    #print image.shape[:2]
    
    size = (h, w, channels) = ( i_height,i_width, 1)
    raw_heat_map = np.zeros(size, np.int8)
    raw_heat_map_float = np.zeros(size)
    raw_heat_map_count = np.zeros(size)
    raw_heat_map_float.fill(0.0)
    raw_heat_map_count.fill(0.0)
    
    w_height=int(i_height*app.window_height_fraction)
    w_width=int(i_width*app.window_width_fraction)
    
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile(app.trained_label_path)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(app.trained_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')        
        
        '''
        # visualize last layer
        #
        #print("ops:", sess.graph.get_operations())
        #sess.close()
        #return
        img = tf.gfile.FastGFile(image_path, 'rb').read()
        feature_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0') #pool_3:0')
        #TODO it seems pool_3:0 is not the last layer, explore again source code
        p = sess.run(feature_tensor, \
                     {'DecodeJpeg/contents:0': img}) 
        #print(len(p[0][0][0])) #2048
        #print(p[0][0][0])
        print(p)
        #threshholding
        
        #for i in range(len(p[0][0][0])):
        #    if (p[0][0][0][i] < .8):
        #        p[0][0][0][i] = 0.0
        
        #plt.imshow(np.reshape(p[0][0][0],[32,64]), interpolation="nearest", cmap="gray")                      
        plt.imshow(np.reshape(p,[32,64]), interpolation="nearest", cmap="gray")                              
        f = '/home_local/shar_sc/'+ 'image.png'
        plt.savefig(f, bbox_inches='tight')
        plt.show()
        #TODO remove these two lines
        sess.close()
        return
        #
        #
        #'''
         
        #slide
        y=0
        while y+w_height <= i_height:
        #while y < i_height:
            x=0    
            while x+w_width <= i_width:        
            #while x < i_width:        
                i_segment = image[y:y+w_height,x:x+w_width]
                #print 'x:', x,'',x+w_width
                #print 'y:', y,'',y+w_height
                
                # write on heat map
                name = app.out_dir + 'temp.jpeg'
                cv2.imwrite(name,i_segment) #TODO donot write file on disk
                
                # Read in the image_data
                image_data = tf.gfile.FastGFile(name, 'rb').read()
                
                #Numpy array
                #i_segment= cv2.resize(i_segment,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
                #np_image_data = np.asarray(i_segment)
                #np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
                #np_final = np.expand_dims(np_image_data,axis=0)
        
                #classify
                predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
                #predictions = sess.run(softmax_tensor,
                #           {'Mul:0': np_final})
                #sess.graph.finalize()  # Graph is read-only after this statement.
        
                prob = process_prob(predictions,label_lines)
                
                #print('predicting probability:', prob)
                
                #size_small = (h, w, channels) = ( w_height,w_width, 1)
                #prob_segment_float = np.zeros(size_small)
                #prob_segment_float.fill(prob)
                
                for k in range(y,y+w_height):
                    for l in range(x,x+w_width):
                        raw_heat_map_float[k][l] += prob
                        raw_heat_map_count[k][l] += 1.0
                
                x+=app.step_size_width            
            y+=app.step_size_height      
            #cv2.imwrite(app.out_dir + 'last_window' + '.jpeg',i_segment) #this overwrites itself in loop. Keep it commentted. Just for debugging.
    sess.close()
    #print(raw_heat_map_float)    
    # normalize probabilities
    for j in range(i_width):
        for i in range(i_height):
            if raw_heat_map_count[i][j] != 0.0:
                raw_heat_map_float[i][j] = raw_heat_map_float[i][j]/raw_heat_map_count[i][j]
                if raw_heat_map_float[i][j] < app.binary_threshold: #TODO workaround for threshold
                    #print 'zeroing: ', raw_heat_map_float[i][j]
                    raw_heat_map_float[i][j] = 0.0
                else:
                    raw_heat_map_float[i][j] = raw_heat_map_float[i][j]*255.0                 
            else:
               if raw_heat_map_float[i][j]!=0.0:
                   print('----------Error---------')
    
    #for j in range(i_width):
    #    for i in range(i_height):
    #        raw_heat_map[i][j] = raw_heat_map_float[i][j]*255.0 #*255, for color and any value below 1.0 is 0 in integer
        
    #make color map from grey scale probability map
    heat_map = cv2.applyColorMap(raw_heat_map_float,cv2.COLORMAP_JET) #cv2.COLORMAP_AUTUMN) #cv2.COLORMAP_OCEAN)
    cv2.imwrite(app.out_dir + 'heat_map' + '.jpeg',heat_map)
    img = cv2.imread(app.out_dir + 'heat_map' + '.jpeg')
    #cv2.threshold(img,200,255,cv2.THRESH_BINARY)#TOZERO)
    im_color = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)# cv2.COLORMAP_JET)
    cv2.imwrite(app.out_dir + 'heat_map_th' + '.jpeg',im_color)
    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(app.out_dir + 'heat_map_thr' + '.jpeg',im_color)

    #print(heat_map)
    return