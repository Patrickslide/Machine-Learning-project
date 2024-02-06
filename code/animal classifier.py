import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


model = load_model('C:\\Users\\patri\\Desktop\\PATRICK\\Università\\Didattica\\Corsi\\Data Science\\Introduction to Machine Learning\\DATA.h5')

# check the model and see if you have to remove the first layer
# if you need to remove the first layer, replace the 0 with 1
# if you need to remove more layers at the end replace the -2
model.summary()

model = keras.Model(model.layers[0].input, model.layers[-2].output)


# get the image paths
def get_image_paths(data_dir):
    out = []
    # for each image in the folder get the path
    for imgs in os.listdir(data_dir):
            out.append(data_dir + imgs)
    return out


img_width = 120
img_height = 120
# extract the features of a given image -- input is the path of the image
def extract_features(inp_img_pth):	
	img = tf.keras.utils.load_img(inp_img_pth, target_size=(img_height, img_height))
	img_array = tf.keras.utils.img_to_array(img)
	img_array = img_array/255.0
	img_array = tf.expand_dims(img_array, 0)
	return model.predict(img_array)[0]



# get the paths for query and gallery images

query_paths = get_image_paths('C:\\Users\\patri\\Desktop\\PATRICK\\Università\\Didattica\\Corsi\\Data Science\\Introduction to Machine Learning\\DATA\\dataset\\validation\\query\\')
print('found %s images for the queries' % (len(query_paths)))

gallery_paths = get_image_paths('C:\\Users\\patri\\Desktop\\PATRICK\\Università\\Didattica\\Corsi\\Data Science\\Introduction to Machine Learning\\DATA\\dataset\\validation\\gallery\\')
print('found %s images in gallery' % (len(gallery_paths)))



# init empty list where we save the gallery features
gallery_features = []

# for each path in the gallery paths extract the feature of the image and add to the gallery_features list
for p in gallery_paths:
	gallery_features.append(extract_features(p))

# init empty list where we save the query features
query_features = []

# for each path in the query paths extract the feature of the image and add to the query_features list
for p in query_paths:
	query_features.append(extract_features(p))


# init empty dict where we'll save the matching between query and gallery images
# keys are the query paths
# values are lists of top10 most similar images	
matching = {}

# for each query image compute the euclidean distance between the given query image and all the gallery images
for query, q in zip(query_features, query_paths):
	# L2 norm - Minkowski distance with r=2
	# compute euclidean distance
	euc_dist = [norm(query-b) for b in gallery_features]
	# get the sorted indeces of gallery images
	ids = np.argsort(euc_dist)[::-1]
	# add to the dictionary a key-value pair
	# key = gallery paths
	# value = sorted list of paths -- top10
	matching[q] = [gallery_paths[ids[-i]] for i in range(1, 11)]


# get some query paths
qtest = list(matching.keys())[:5]

# plot some query images with top10
for queries in qtest:
	plt.figure(figsize=(20, 20))
	qpth = queries
	qimg = plt.imread(qpth)
	ax = plt.subplot(3,4,1)
	plt.imshow(qimg)
	plt.title('query image')
	for i in range(10):
	    pth = matching[qpth][i]
	    img = plt.imread(pth)
	    ax = plt.subplot(3,4,i+2)
	    plt.imshow(img)
	    plt.axis("off")
	plt.show()


# since we have not to send the complete path but only the image names, it is necessary to edit the dictionary
# we remove the initial path from both keys and list values

# init empty dictionary to be sent
out = {}

for match in matching:
	# get query image name
	# LAST-THING-BEFORE-IMAGE-NAME -- e.g. 'query', ')' (if animal name), etc.
	q_path = match.split('LAST-THING-BEFORE-IMAGE-NAME/')[1]
	# get gallery image names
	g_paths = [g.split('LAST-THING-BEFORE-IMAGE-NAME/')[1] for g in matching[match]]
	# add to the dictionary the key-value pair
	# key = query name
	# value = list of gallery names -- top10
	out[q_path] = g_paths


########################################
#			   		 #
#	 WAIT FOR THIS PART!!! 	 #
#			 		 #
########################################


# send everything

import requests
import json

# the url probably is different
def submit(results, url="http://coruscant.disi.unitn.it:3001/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


# init dictionary to be sent
mydata = dict()
# add name of the group
mydata['OMPF'] = "request"

# add the final dictionary of matching to mydata
mydata["images"] = out

# send
submit(mydata)

	
	
	
	
	

