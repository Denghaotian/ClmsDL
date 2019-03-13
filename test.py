# Qingbo/Haotian Mar 8,2019

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from loaddata import load_dataset

def main(_):
    #=============Load label====================
    #Load labels
    label_lst=[]
    rs = os.path.exists(FLAGS.output_labels)
    if rs==True:
        file_handler =open(FLAGS.output_labels,mode='r')
        contents = file_handler.readlines()
        for name in contents:
            name = name.strip('\n')
            label_lst.append(name)
        file_handler.close()
    print(label_lst)

    #Prepare input data
    classes = os.listdir(FLAGS.test_path )
    num_classes = len(classes)
    print("number classes is ",num_classes)

    # load all the training and validation images and labels into memory using openCV and use that during training
    test_data, test_label, _, cls_batch, categories = load_dataset(FLAGS.test_path, FLAGS.img_size, classes)

    with tf.Session() as sess:
        ## Let us restore the saved model 
        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph(FLAGS.trained_model)
        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))

        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network labels_pred is the tensor that is the prediction of the network
        labels_pred = graph.get_tensor_by_name("labels_pred:0")

        ## Let's feed the images to the input placeholders
        data_placeholder= graph.get_tensor_by_name("data_placeholder:0") 
        label_placeholder = graph.get_tensor_by_name("label_placeholder:0") 

        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {data_placeholder: test_data, label_placeholder: test_label}
        labels_pred_cls = tf.argmax(labels_pred, axis=1)
        labels_true_cls = tf.argmax(test_label, axis=1)
        #================for debug==============
        # result=sess.run(y_pred, feed_dict=feed_dict_testing)
        # result=sess.run(labels_pred_cls, feed_dict=feed_dict_testing)
        # result=sess.run(labels_true_cls, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_A, probability_of_B]
        # print(result)
        #================for debug==============
        correct_prediction = tf.equal(labels_pred_cls, labels_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict=feed_dict_testing)
        msg = "testing Accuracy: {0:>6.1%}"
        print(msg.format(acc))


if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("test_path", "data_cat/testing_data", "path of testing data")
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_string("output_labels", "trained_model/output_labels.txt", "store the labels")
    flags.DEFINE_string("trained_model", "trained_model/bully_action.meta", "meta graph")
    flags.DEFINE_string("checkpoint", "./trained_model/", "checkpoint")

    tf.app.run()