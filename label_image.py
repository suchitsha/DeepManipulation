# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (10, 110, 330, 80),
#  'transitions': []}
### end of header
import tensorflow as tf, sys

def init(self):
    pass

def execute(self):
    print "execute of %s called!" % self.name
    
    #label image
    image_path = app.test_image_path #sys.argv[1]
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
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
    return            
    #return dict(exit="all_processed")### end of gui script