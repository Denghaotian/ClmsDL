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

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    x_batch, y_true_batch, _, cls_batch, categories = load_dataset(FLAGS.test_path, FLAGS.img_size, classes)

    with tf.Session() as sess
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


if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("test_path", "data_cat/testing_data", "path of testing data")
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_string("output_labels", "trained_model/output_labels.txt", "store the labels")
    flags.DEFINE_string("trained_model", "trained_model/dogs-cats-model.meta", "meta graph")
    flags.DEFINE_string("checkpoint", "./trained_model/", "checkpoint")

    tf.app.run()