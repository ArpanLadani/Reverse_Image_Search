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


def main():
    with st.sidebar:
        st.image("logo.jfif")
        st.title("Reverse Image Search")
        st.info("This app helps us to find images that are similar to the uploaded image by you.")
        choice = st.radio("Navigation", ["Upload Picture", "About Us" ,"Cancel"])

    if choice == "Upload Picture":
        st.title('Upload your Image you want to search')
        file = st.file_uploader("Upload your image here")
        if file:
            st.image(file)

    if choice == "About Us":
        st.title('I am Arpan Ladani')
        st.text('Email: ladaniarpan@gmail.com')

    if choice == "Cancel":
        pass

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass