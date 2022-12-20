# %%

# GAN PROJECT

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# %%

# Arguments

# preprocessing arguments
dir = "balaban_resim_cropped" # directory of the images
img_size = 32 # img_size x img_size images

# sample boost arguments
sample_boost_size = 46 * 5 # how many times to sample from the dataset
rotation_range = 10 # rotation range for image augmentation
width_shift_range = 0.1 # width shift range for image augmentation
height_shift_range = 0.1 # height shift range for image augmentation
zoom_range = 0.2 # zoom range for image augmentation
horizontal_flip = True # horizontal flip for image augmentation
vertical_flip = True # vertical flip for image augmentation

# filter arguments

# color space conversion for the filters
grayscale = False

# noise reduction for the filters # TODO: add filters
median_filter = True
median_filter_size = 3 
gaussian_filter = False 
gaussian_filter_size = 3 
biliteral_filter = False 
biliteral_filter_size = 3 
biliteral_filter_sigma_color = 25 
biliteral_filter_sigma_spatial = 25 


# model arguments
batch_size = 128 # batch size

# %% 

# Load the dataset and preprocess it

files = os.listdir(dir)
# open images from the folder
images = []
for file in files:
    img = tf.keras.utils.load_img(dir + "/" + file)

    if grayscale:
        img = img.convert(mode="L")
    
    if median_filter:
        img = tf.image.median_filter2d(img, median_filter_size)
    
    # resize images
    img = img.resize((img_size, img_size))
    
    # convert to numpy array
    img = tf.keras.utils.img_to_array(img)

    # append to the list
    images.append(img)

# %%
# show image
plt.imshow(images[1].astype(np.uint8), cmap="gray")
plt.show()

print("Number of images: ", len(images))

# %%
# image augmentation (sampele boost)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip,
    zoom_range=zoom_range,
)

imageIterator = datagen.flow(np.array(images), batch_size=1)

# append augmented images to the list
for i in range(sample_boost_size):
    images.extend(imageIterator.next())

print("Number of images boosted: ", len(images))
# %%
# show last 5 generated images

for i in range(1, 5):
    plt.imshow(images[-i].astype(np.uint8))
    plt.show()

# %%

# 