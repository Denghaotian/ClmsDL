# Qingbo/Haotian Mar 8,2019
# from __future__ import absolute_import  
# from __future__ import division  
# from __future__ import print_function    
import loaddata
from mymodel import *
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
    mygraph = tf.Graph()
    with mygraph.as_default():
        #1) Define some data & labbel placeholder.
        ## data
        data_placeholder = tf.placeholder(tf.float32, 
            shape=[None, FLAGS.img_size,FLAGS.img_size,FLAGS.img_depth], 
            name='data_placeholder')
        ## labels
        label_placeholder = tf.placeholder(tf.float32, 
            shape=[None, FLAGS.class_number], name='label_placeholder')
        # label_index = tf.argmax(label_placeholder, dimension=1)
        label_index = tf.argmax(label_placeholder, axis=1)
        # print("label_index is :", label_in)

        #2) initilize the weight matrices and bias vectors 
        coefficients = define_coefficients(filter_size=FLAGS.filter_size, 
            img_depth=FLAGS.img_depth, filter_depth1=FLAGS.filter_depth1,
            filter_depth2=FLAGS.filter_depth2,
            filter_depth3=FLAGS.filter_depth3,
            flatten_num=FLAGS.flatten_num, fc_depth=FLAGS.fc_depth,
            class_number=FLAGS.class_number)

        #3. construct the CNN model
        logits = lainet(data_placeholder, coefficients)

        #4. calculate the cross entropy between the logits and actual labels
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits= 
                        logits, labels=tf.stop_gradient(label_placeholder))
        #兼容tensorflow1.5之前
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= 
        #                 logits, labels=label_placeholder)
        cost = tf.reduce_mean(cross_entropy)

        #5. use optimizer to calculate the gradients of the loss function 
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) \
                    .minimize(cost)

        # Predictions for the training, validation, and test data.
        labels_pred = tf.nn.softmax(logits,name='y_pred')
        class_pred= tf.argmax(labels_pred, axis=1)
        #class_pred= tf.argmax(labels_pred, dimension=1)
        correct_pred = tf.equal(class_pred, label_index)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session(graph=mygraph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(FLAGS.iteration_steps):
            train_data_batch, train_label_batch, _, train_cls_batch = \
                input_data.train.next_batch(FLAGS.batch_size)
            val_data_batch, val_label_batch, _, val_cls_batch = \
                input_data.valid.next_batch(FLAGS.batch_size)
            
            feed_dict_train = {data_placeholder: train_data_batch,
                            label_placeholder: train_label_batch}
            feed_dict_val = {data_placeholder: val_data_batch,
                                label_placeholder: val_label_batch}
            sess.run(optimizer, feed_dict=feed_dict_train)

            if i % int(input_data.train.num_examples/FLAGS.batch_size) == 0: 
                val_loss = sess.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(input_data.train.num_examples/FLAGS.batch_size))    
                train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
                val_acc = sess.run(accuracy, feed_dict=feed_dict_val)
                msg = ("Training Epoch {0} --- Training Accuracy: {1:>6.1%}," 
                    "Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}")
                print(msg.format(epoch + 1, train_acc, val_acc, val_loss))

                #save the result
                # saver.save(session, 'trained_model/bully-model') 
                saver.save(sess, 'trained_model/dogs-cats-model') 


if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    # ?% of the data will be used for validation
    flags.DEFINE_float('validation_size', 0.2, 'validation size.')
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_integer('iteration_steps', 4000, 'Number of epochs to run trainer.')
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
