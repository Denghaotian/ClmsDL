# Qingbo/Haotian Mar 8,2019
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function    
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
    print(FLAGS.learning_rate)
    print(FLAGS.epoch_number)
    #prepare the training dataset, load data
    #to be done

    #create a dataflow graph
    graph = tf.Graph()
    with graph.as_default():
        #1) Transform the training data to  tensorflow type.
        #2) initilize the weight matrices and bias vectors 
        #3. construct the model
        logits = model(tf_train_dataset, variables)
        #4. calculate the softmax cross entropy between the logits and actual labels
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        #5. use optimizer to calculate the gradients of the loss function 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))


    with tf.Session(graph=graph) as sess:
        #
        #store the trained model
        saver = tf.train.Saver()

if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    # ?% of the data will be used for validation
    flags.DEFINE_float('validation_size', 0.2, 'validation size.')
    flags.DEFINE_integer('epoch_number', 4000, 'Number of epochs to run trainer.')
    flags.DEFINE_integer('batch_size', 32, 'Number of batch size.')
    flags.DEFINE_string("train_path", "data_cat/training_data", "path of training data")

    tf.app.run()
