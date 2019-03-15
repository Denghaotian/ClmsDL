import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# from loaddata import load_dataset
from tensorflow.python.platform import flags  

def main(_):
    #Load labels
    # label_lst=[]
    # # rs = os.path.exists(FLAGS.output_labels)
    # rs = os.path.exists("trained_model/output_labels.txt")
    # if rs==True:
    #     file_handler =open("trained_model/output_labels.txt",mode='r')
    #     contents = file_handler.readlines()
    #     for name in contents:
    #         name = name.strip('\n')
    #         label_lst.append(name)
    #     file_handler.close()
    # print(label_lst)

    #==============================================================
    # Prepare input data
    classes = os.listdir(FLAGS.train_path)
    classes.sort()
    num_classes = len(classes)
    print("number classes is ",num_classes)

    # # We shall load all the training and validation images and labels into memory using openCV and use that during training
    # x_batch, y_true_batch, _, cls_batch, categories = load_dataset(train_path, FLAGS.img_size, classes)
    # print(y_true_batch)
    #==============================================================

    #======================read the image========================================
    # First, pass the path of the image
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # image_path=sys.argv[1] 
    # filename = dir_path +'/' +image_path
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(FLAGS.img_file)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (FLAGS.img_size, FLAGS.img_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    input_img = images.reshape(1, FLAGS.img_size,FLAGS.img_size,FLAGS.num_channels)
    #========================read the image======================================

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
        # y_test_images = np.zeros((1, len(os.listdir('data_bully/train_data')))) 
        label_false = np.zeros((1, len(classes))) 

        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {data_placeholder: input_img, label_placeholder: label_false}
        result=sess.run(labels_pred, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        print(result)
        #print(label_lst)
        class_pred= tf.argmax(result, axis=1)
        label_index=sess.run(class_pred)
        print(label_index)
        print(classes[label_index[0]])
        # print(label_lst[label_index[0]])

    # img = Image.open("data_cat/testing_data/1018.jpg")
    plt.figure(figsize=(6,4))
    plt.subplot(2,1,1)
    # print(input_img[0])
    # plt.imshow(input_img)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2,1,2)
    # plt.figure()
    plt.barh([0, 1], result[0], alpha=0.5)
    plt.yticks([0, 1], classes)
    #plt.yticks([0, 1], label_lst)
    plt.xlabel('Probability')
    plt.xlim(0,1.01)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #set some superparameters which can reset befor run
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("train_path", "data_bully/training_data", "path of testing data")
    flags.DEFINE_string("img_file", "data_bully/testing_data/gossiping/gossiping0001.jpg", "path of testing data")
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_integer('num_channels', 3, 'image channel number.')
    # flags.DEFINE_string("output_labels", "trained_model/output_labels.txt", "store the labels")
    flags.DEFINE_string("trained_model", "trained_model/bully_action.meta", "meta graph")
    flags.DEFINE_string("checkpoint", "./trained_model/", "checkpoint")

    tf.app.run()