# Qingbo/Haotian Mar 8,2019
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function    
import loaddata
import mymodel
import sys  
import os
import tensorflow as tf
from tensorflow.python.platform import flags  

import time
from datetime import timedelta
import math
import random
import numpy as np


def main(_):
    print("train_path is :", FLAGS.train_path)
    print("validation percent is :", FLAGS.validation_size)
    #prepare the training dataset & load data
    classes = os.listdir(FLAGS.train_path)
    flags.DEFINE_integer('class_number', len(classes), 'classes number.')
    # class_number= len(classes)
    print("class number is:", FLAGS.class_number)
    input_data = loaddata.read_dataset(FLAGS.train_path, FLAGS.img_size, 
                                      classes, FLAGS.validation_size)
    print("******The traning data have been loaded**********")
    print("Number of files in Training-set:  \t{}" 
         .format(len(input_data.train.labels)))
    print("Number of files in Validation-set:\t{}" 
        .format(len(input_data.valid.labels)))

    #create a dataflow graph
    graph = tf.Graph()
    with graph.as_default():
        #1) Define some data & labbel placeholder.
        ## data
        data_placeholder = tf.placeholder(tf.float32, 
            shape=[None, FLAGS.img_size,FLAGS.img_size,FLAGS.img_depth], 
            name='data_placeholder')
        ## labels
        label_placeholder = tf.placeholder(tf.float32, 
            shape=[None, FLAGS.class_number], name='label_placeholder')
        # label_number = tf.argmax(label_placeholder, dimension=1)
        label_number = tf.argmax(label_placeholder, axis=1)
        # print("label_number is :", label_number)

        #2) initilize the weight matrices and bias vectors 
        coefficients = define_coefficients(filter_size=FLAGS.filter_size, 
            img_depth=FLAGS.img_depth, filter_depth1=FLAGS.filter_depth1,
            fileter_depth2=FLAGS.fileter_depth2,
            fileter_depth3=FLAGS.fileter_depth3,
            flatten_num=FLAGS.flatten_num, class_number=FLAGS.class_number)

        #3. construct the CNN model
        logits = lainet(data_placeholder, coefficients)

        #4. calculate the softmax cross entropy between the logits and actual labels
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        #5. use optimizer to calculate the gradients of the loss function 
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Predictions for the training, validation, and test data.
        # train_prediction = tf.nn.softmax(logits)
        # test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))

    # with tf.Session(graph=graph) as sess:
        # saver = tf.train.Saver()
        #
        #store the trained model

if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    # ?% of the data will be used for validation
    flags.DEFINE_float('validation_size', 0.2, 'validation size.')
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_integer('epoch_number', 4000, 'Number of epochs to run trainer.')
    flags.DEFINE_integer('batch_size', 32, 'Number of batch size.')
    flags.DEFINE_string("train_path", "data_cat/training_data", "path of training data")

    
    flags.DEFINE_integer('img_depth', 3, 'Number of channels.')
    flags.DEFINE_integer('filter_size', 3, 'filter size.')
    flags.DEFINE_integer('filter_depth1', 32, 'filter depth for conv1.')
    flags.DEFINE_integer('filter_depth2', 32, 'filter depth for conv2.')
    flags.DEFINE_integer('filter_depth3', 64, 'filter depth for conv3.')
    flags.DEFINE_integer('pooling_num', 3, 'Number of pooling')
    flags.DEFINE_integer('flatten_num', pow(FLAGS.img_size//
        pow(2,FLAGS.pooling_num),2)*FLAGS.filter_depth3, 
        'Number of features after flattern')
    print("flatten_nub is", FLAGS.flatten_num)
    flags.DEFINE_integer('fc_depth', 128, 'fully connected layer depth.')


    tf.app.run()
