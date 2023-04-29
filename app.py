import streamlit as st
import cv2 as cv
import pandas as pd
import requests
import os
import numpy as np
from numpy.linalg import norm
import joblib as pickle
from tqdm import tqdm
import os
import PIL
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from PIL import Image
import io
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

def load_dataset():
    img_size = 224
    root_dir = 'D:/sem8/Scaledge/Reverse_image/Reverse_Image_Search/caltech-101/101_ObjectCategories'

    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input) #setting pipeline

    datagen = img_gen.flow_from_directory(root_dir,
                                            target_size=(img_size, img_size),
                                            batch_size=64,
                                            class_mode=None,
                                            shuffle=False)
    global filenames
    filenames = [root_dir + '/' + s for s in datagen.filenames]

def load_model_inmemory():
    model = tf.keras.models.load_model('models/model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# def scale(image):
#     image = image/255
#     input_maps = tf.image.resize_images(input_maps, (image_size[0], image_size[1]))
#     return tf.image.resize(image, tf.constant[224,224])

# def load_features():
#     features = np.load('features.npy')

# def nearest_neighbor():
#     neighbors = NearestNeighbors(n_neighbors=5,algorithm='ball_tree',metric='euclidean')
#     neighbors.fit(features)

def decode_image(image):
    # img = scale(image)
    expanded_img_array = np.expand_dims(image, axis = 0)
    expanded_img_array = np.copy(expanded_img_array)
    preprocessed_img = preprocess_input(expanded_img_array)
    return preprocessed_img
    # test_img_features = model.predict(preprocessed_img, batch_size=1)
    # return test_img_features


def similar_images(indices):
    # plt.figure(figsize=(15,10), facecolor='white')
    # plotnumber = 1    
    
    # st.write(
    #     f'<style>.stImage {{ max-height: 300px; object-fit: contain; }}</style>',
    #     unsafe_allow_html=True,
    # )
    return st.image(mpimg.imread(filenames[indices]), use_column_width=True)
    # for index in indices:
    #     if plotnumber<=len(indices) :
    #         ax = plt.subplot(2,4,plotnumber)
    #         # plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')  
    #         st.text(plotnumber)
    #         st.image(mpimg.imread(filenames[index]),width= 200)
            
    #         print(filenames[index])          
    #         plotnumber+=1
    # return plt.tight_layout()

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.write(
        '''<style>
        .element-container.css-rton5x.e1tzin5v3 {
            width:60px;
        }</style>''',
        unsafe_allow_html=True,
    )
    st.image("scaledge_logo.jpg", use_column_width = True,channels="RGB")

    # with st.sidebar:
    #     st.image("logo.jfif", use_column_width = True)
    #     st.title("Reverse Image Search")
    #     st.info("This app helps us to find images that are similar to the uploaded image by you.")
    #     choice = st.radio("Navigation", ["Upload Picture", "About Us" ,"Cancel"])
    
    # with st.sidebar:
    #     lottie_search = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_jo7huq2d.json")
    #     st_lottie(lottie_search,key= "search",height = 300, width = 300,speed= 0.75)
    #     selected = option_menu(
    #         menu_title="Reverse Image Search",
    #         options=["About Us", "Upload Picture", "Contact"],
    #         icons=["info-circle", "cloud-arrow-up-fill", "envelope"],
    #         default_index= 0
    #     )
    with st.sidebar:
        lottie_search = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_bo8vqwyw.json",)
        st_lottie(lottie_search,key= "search",height = 280, width = 280,speed= 0.50)
        selected = option_menu(
            menu_title="Reverse Image Search",
            menu_icon= None,
            options=["About Us", "Upload Picture", "Contact"],
            icons=["info-circle", "cloud-arrow-up-fill", "envelope"],
            default_index= 0,
            orientation= "vertical"
        )

    if selected == "Upload Picture":
        # st.title('Upload your Image you want to search')
        image_file_buffer = st.file_uploader("Upload your image you want to search")
        # with st.spinner ('Loading dataset into Memory...'):
        #     dataset = load_dataset()
        
        if image_file_buffer is not None:
            bytes_data = image_file_buffer.getvalue()
            st.image(bytes_data)
            s_image = io.BytesIO(bytes_data)
            input_shape = (224, 224, 3)
            img = image.load_img(s_image, target_size=(input_shape[0], input_shape[1]))
            # st.image(image_file_buffer, caption="Selected Image", use_column_width = True)
            # image_open = Image.open(image_file_buffer)
            # img_array_open = np.array(image_file_buffer)
            # with st.spinner ('Loading Model into Memory...'):
            #     model = load_model()
            # with st.spinner ('Loading Features into Memory...'):
            #     features = np.load('features.npy')
            
            # img_path = 'aeroplane.jfif'
            # input_shape = (224, 224, 3)
            # system_img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
            preprocessed_img = decode_image(img)
            with st.spinner('Predicting the image'):
                test_img_features = model.predict(preprocessed_img, batch_size=1)
                neighbors = NearestNeighbors(n_neighbors=5,algorithm='ball_tree',metric='euclidean')
                neighbors.fit(features)
                _, indices = neighbors.kneighbors(test_img_features)
                print(indices)
                
                print(indices.shape)
                # plt.imshow(mpimg.imread(image_open), interpolation='lanczos')
                # plt.xlabel(image_open.split('.')[0] + '_Original Image',fontsize=20)
                # plt.show()
                print('Predictions images')
                col1,col2,col3,col4,col5 = st.columns(5)
                with col1:
                    similar_images(indices[0][0])
                with col2:
                    similar_images(indices[0][1])
                with col3:
                    similar_images(indices[0][2])
                with col4:
                    similar_images(indices[0][3])
                with col5:
                    similar_images(indices[0][4])

            print("Done")
            
    if selected == "About Us":
        st.title('')
        st.text('--> Reverse image search is a search engine technology that takes an image file as \n input query and returns results related to the image.')

    if selected == "Contact":
        st.text('Email:  ladaniarpan@gmail.com')
        st.text('Github Repository:  https://github.com/ArpanLadani/Reverse_Image_Search')
        st.text_input('If any query type here we will contact you soon')
        pass


if __name__ == "__main__":
    try:
        with st.spinner ('Loading dataset into Memory...'):
            dataset = load_dataset()
        with st.spinner ('Loading Model into Memory...'):
            model = load_model_inmemory()
        with st.spinner ('Loading Features into Memory...'):
            features = np.load('features.npy')
        main()
    except SystemExit:
        pass