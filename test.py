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




if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("test_path", "data_cat/testing_data", "path of testing data")
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_string("output_labels", "trained_model/output_labels.txt", "store the labels")

    tf.app.run()