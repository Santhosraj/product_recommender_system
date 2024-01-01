# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:32:14 2023

@author: SanthosRaj
"""
import tensorflow
import numpy as np
import pickle
from keras.preprocessing import image
import sklearn,keras
from numpy import linalg as LA
from keras.applications.resnet import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors

import cv2


feature_list = np.array(pickle.load(open("D:/Santhosraj Machine learning/spyder/embeddings.pkl","rb")))
filenames = pickle.load(open("D:/Santhosraj Machine learning/spyder/filenames.pkl","rb"))



model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([model,
                                 GlobalMaxPooling2D(),
                                 ])

img = keras.utils.load_img("D:/Santhosraj Machine learning/spyder/fashion_sample_img/watch.jpg",target_size=(224,224))

img = keras.utils.img_to_array(img)
img= np.expand_dims(img, axis=0)
preprocessed_img = preprocess_input(img)

result = model.predict(preprocessed_img).flatten()
normalized_result = result/LA.norm(result)


neighbors = NearestNeighbors(n_neighbors=5,algorithm="brute",metric="euclidean")
neighbors.fit(feature_list)

distances , indices = neighbors.kneighbors([normalized_result])
print(indices)

for file  in indices[0][1:6]:
  print(filenames[file])
 
