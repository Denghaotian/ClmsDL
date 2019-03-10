import tensorflow as tf

def flatten_the_layer(array):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

def define_coefficients(filter_size, img_depth, filter_depth1, fileter_depth2
                  filter_depth3,flatten_num,class_number):
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, img_depth, filter_depth1], stddev=0.05))
    b1 = tf.Variable(tf.constant(0.05, shape=[filter_depth1])
    # b1 = tf.Variable(tf.zeros([filter_depth1])
    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth1, filter_depth2], stddev=0.05))
    b2 = tf.Variable(tf.constant(0.05, shape=[filter_depth2])
    w3 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth2, filter_depth3], stddev=0.05))
    b3 = tf.Variable(tf.constant(0.05, shape=[filter_depth3])
    w4 = tf.Variable(tf.truncated_normal([flatten_num, fc_depth], stddev=0.05))
    b4 = tf.Variable(tf.constant(0.05, shape=[fc_depth])
    w5 = tf.Variable(tf.truncated_normal([fc_depth,class_number], stddev=0.05))
    b5 = tf.Variable(tf.constant(0.05, shape=[class_number])
    coefficients = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return coefficients