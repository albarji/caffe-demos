"""
Classifies a given image into one out of 1000 ImageNet categories (http://www.image-net.org)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe

# Check arguments
if len(sys.argv) < 2:
    print("Arguments: <image_to_classify>")
    sys.exit(0)

# Set the right path to your model definition file and pretrained model weights
MODEL_FILE = 'deploy.prototxt'
PRETRAINED = 'bvlc_reference_caffenet.caffemodel'

# Set caffe to CPU mode
caffe.set_mode_cpu()
# Load pretrained model from file
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
               mean=np.load('ilsvrc_2012_mean.npy').mean(1).mean(1),
               channel_swap=(2,1,0),
               raw_scale=255,
               image_dims=(256, 256))
               
# Load labels files
with open('synset_words.txt', 'rb') as labelsfile:
    labels = labelsfile.readlines()
    
# Load an image to classify
IMAGE_FILE = sys.argv[1]
input_image = caffe.io.load_image(IMAGE_FILE)

# Classify image
likelihoods = net.predict([input_image])
# Print most likely classes
print '\nMost likely classes:'
mostlikely = sorted(zip(labels, likelihoods[0]), key=lambda x:x[1], reverse=True)
for i in range(5):
    print(mostlikely[i][1], mostlikely[i][0])

