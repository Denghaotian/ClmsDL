from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import argparse
import sys,os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import errno 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def run_graph(src, dest):
#def run_graph(src, dest, labels, input_layer_name, output_layer_name, num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        # predictions  will contain a two-dimensional array, where one
        # dimension represents the input image count, and the other has
        # predictions per class
        i=0
        #with open('submit.csv','w') as outfile:
        for f in os.listdir(src):
#             print(f)
            new_f=os.path.splitext(f)[0]
#             print(new_f)
            im=Image.open(os.path.join(src,f))
            #print("im is ",im)
            img=im.convert('RGB')
#             print(os.path.join(dest,new_f+'.jpg'))
            img.save(os.path.join(dest,new_f+'.jpg'))
            i+=1
        print("done")

# src=os.path.join('./data_bully/','train_data/gossiping/')
# src=os.path.join('./data_bully/','train_data/isolation/')
# src=os.path.join('./data_bully/','train_data/laughing/')
# src=os.path.join('./data_bully/','train_data/pullinghair/')
# src=os.path.join('./data_bully/','train_data/punching/')
# src=os.path.join('./data_bully/','train_data/quarrel/')
# src=os.path.join('./data_bully/','train_data/slapping/')
# src=os.path.join('./data_bully/','train_data/stabbing/')
# src=os.path.join('./data_bully/','train_data/strangle/')

# src=os.path.join('./data_bully/','train_data_09/gossiping/')
# src=os.path.join('./data_bully/','train_data_09/isolation/')
# src=os.path.join('./data_bully/','train_data_09/laughing/')
# src=os.path.join('./data_bully/','train_data_09/pullinghair/')
# src=os.path.join('./data_bully/','train_data_09/punching/')
# src=os.path.join('./data_bully/','train_data_09/quarrel/')
# src=os.path.join('./data_bully/','train_data_09/slapping/')
# src=os.path.join('./data_bully/','train_data_09/stabbing/')
# src=os.path.join('./data_bully/','train_data_09/strangle/')
#src=os.path.join('./data_bully/','train_data_09/nobully/')

#print(src)
# dest=os.path.join('./data_bully/','train_data/gossiping/')
# dest=os.path.join('./data_bully/','train_data/isolation/')
# dest=os.path.join('./data_bully/','train_data/laughing/')
# dest=os.path.join('./data_bully/','train_data/pullinghair/')
# dest=os.path.join('./data_bully/','train_data/punching/')
# dest=os.path.join('./data_bully/','train_data/quarrel/')
# dest=os.path.join('./data_bully/','train_data/slapping/')
# dest=os.path.join('./data_bully/','train_data/stabbing/')
# dest=os.path.join('./data_bully/','train_data/strangle/')
src=os.path.join('./data_bully/','test_data/gossiping')
dest=os.path.join('./data_bully/','test_data2/gossiping')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/isolation')
dest=os.path.join('./data_bully/','test_data2/isolation')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/laughing')
dest=os.path.join('./data_bully/','test_data2/laughing')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/pullinghair')
dest=os.path.join('./data_bully/','test_data2/pullinghair')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/punching')
dest=os.path.join('./data_bully/','test_data2/punching')
mkdir_p(dest)
run_graph(src,dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/quarrel')
dest=os.path.join('./data_bully/','test_data2/quarrel')
mkdir_p(dest)
src=os.path.join('./data_bully/','test_data/slapping')
dest=os.path.join('./data_bully/','test_data2/slapping')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/stabbing')
dest=os.path.join('./data_bully/','test_data2/stabbing')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/strangle')
dest=os.path.join('./data_bully/','test_data2/strangle')
mkdir_p(dest)
run_graph(src,dest)
src=os.path.join('./data_bully/','test_data/nobully')
dest=os.path.join('./data_bully/','test_data2/nobully')
mkdir_p(dest)
run_graph(src,dest)

