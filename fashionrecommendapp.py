# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 22:32:49 2023

@author: SanthosRaj
"""
import streamlit as st
import os
from PIL import Image
import tensorflow
import numpy as np
import pickle
from keras.preprocessing import image
import sklearn,keras
from numpy import linalg as LA
from keras.applications.resnet import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors


model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([model,
                                 GlobalMaxPooling2D(),
                                 ])


feature_list = np.array(pickle.load(open("D:/Santhosraj Machine learning/spyder/embeddings.pkl","rb")))
filenames = pickle.load(open("D:/Santhosraj Machine learning/spyder/filenames.pkl","rb"))



st.title("Product Recommender")

def save_file(uploaded_file):
   try:
       with open(os.path.join("uploads",uploaded_file.name),"wb") as f:
           f.write(uploaded_file.getbuffer())
       return 1
   except:
       return 0

def feature_extraction(filepath,model):
    img = keras.utils.load_img(filepath, target_size=(224, 224))

    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(img)

    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / LA.norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices
#file upload


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
      if save_file(uploaded_file):
          #displaying the file
          display_image = Image.open(uploaded_file)
          st.image(display_image)
          features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
          print(features)
          st.text(features)
          indices = recommend(features,feature_list)

          col1,col2,col3,col4,col5 = st.columns(5)

          with col1:
              st.image(filenames[indices[0][0]])

      else:
          st.header("An error occured in file upload")


