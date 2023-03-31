#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import imageio
import numpy as np


# 

# In[27]:


# Step 1: Load the trained model
model = tf.keras.models.load_model('models.h5')


# In[27]:





# In[28]:


# Step 2: Define a function to get image paths for a given word
def get_image_paths(word):
    image_dir = 'images'
    word_dir = os.path.join(image_dir, word)
    image_paths = []
    for root, dirs, files in os.walk(word_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths


# In[29]:


# Step 3: Set up image generators for data augmentation
datagen = ImageDataGenerator(rescale=1./255)


# In[30]:


# Step 4: Use the model to predict the class of each image
def predict_images(model, image_paths):
    predictions = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array)
        predicted_class = tf.argmax(prediction[0]).numpy()
        predictions.append(predicted_class)
    return predictions


# In[31]:


def get_predicted_image_paths(word):
    image_paths = get_image_paths(word)
    print(f'Number of image_paths: {len(image_paths)}')
    predictions = predict_images(model, image_paths)
    print(f'Predictions: {predictions}')
    predicted_image_paths = []
    for i, prediction in enumerate(predictions):
        if prediction == 1: # Replace 1 with the class label that corresponds to the word
            predicted_image_paths.append(image_paths[i])
    print(f'Number of predicted_image_paths: {len(predicted_image_paths)}')
    return predicted_image_paths


# In[32]:





# In[32]:





# In[37]:


# Step 6: Use imageio to create an mp4 from the list of images
def create_gif(word):
    predicted_image_paths = get_predicted_image_paths(word)
    images = []
    for path in predicted_image_paths:
        img = imageio.imread(path)
        images.append(img)
    mp4_path = f'gif/{word}.mp4'
    imageio.mimsave(mp4_path, images)
    # Play the mp4 file in OpenCV
    cap = cv2.VideoCapture(mp4_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# In[39]:


file = input("enter name: ")
create_gif(file)
cap = cv2.VideoCapture(f'gif/{file}.mp4') # Replace 'filename.mp4' with your video file path
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




