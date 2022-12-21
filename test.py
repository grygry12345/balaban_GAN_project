# %%

# GAN PROJECT

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import glob
import imageio
import math
from IPython import display
from PIL import Image

# %%

# find number of images in the folder
num_img = len(os.listdir("balaban_resim_cropped"))

# %%
# Arguments

# preprocessing arguments
DIR = "balaban_resim_cropped"  # directory of the images
IMG_SIZE = 16  # img_size x img_size images

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100

# if sample boost is selected add these arguments
SAMPLE_BOOST_SIZE = num_img * 2
ROTATION_RANGE = 25
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.6
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False



# filter arguments

# color space conversion for the filters
GRAYSCALE = False

KSIZE = 3  # kernel size for the filters
# noise reduction for the filters (exclusive arguments)
MEDIAN_FILTER = False

# if gausian filter is selected add this argument
GAUSIAN_FILTER = False
SIGMAX = 0

# if biliteral filter is selected add this argument
BILITERAL_FILTER = False
D = 9  # diameter of each pixel neighborhood that is used during filtering
SIGMACOLOR = 75
SIGMASPACE = 75

# %%
# model arguments
NOISE_VECTOR_SIZE = 64  # noise dimension
BATCH_SIZE = 64  # batch size
EPOCHS = 10000 
BUFFER_SIZE = 1000  # buffer size for shuffling the dataset
LR_RATE = 0.0002  # learning rate

# %%

# Arguments check
if MEDIAN_FILTER and GAUSIAN_FILTER or MEDIAN_FILTER and BILITERAL_FILTER or GAUSIAN_FILTER and BILITERAL_FILTER:
    raise Exception("Only one filter can be selected at a time!")
# %%

# Load the dataset and preprocess it

files = os.listdir(DIR)
# open images from the folder
images = []
for file in files:
    img = tf.keras.utils.load_img(DIR + "/" + file)

    if GRAYSCALE:
        img = img.convert("L")

    # resize images
    resized_img = img.resize((IMG_SIZE, IMG_SIZE))

    # convert to numpy array
    img_array = tf.keras.utils.img_to_array(resized_img)

    if MEDIAN_FILTER:
        img_array = cv2.medianBlur(img_array, ksize=KSIZE)
    elif GAUSIAN_FILTER:
        img_array = cv2.GaussianBlur(
            img_array, ksize=(KSIZE, KSIZE), sigmaX=SIGMAX)
    elif BILITERAL_FILTER:
        img_array = cv2.bilateralFilter(
            img_array, d=9, sigmaColor=75, sigmaSpace=75)

    # append to the list
    images.append(img_array)

# %%
# show image
plt.imshow(images[1].astype(np.uint8), cmap="gray")
plt.show()

print("Number of images: ", len(images))

# %%
# image augmentation (sample boost)

# if rgayscale reshpae the images for 4d tensor
if GRAYSCALE:
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    image = images.tolist()


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    vertical_flip=VERTICAL_FLIP,
    zoom_range=ZOOM_RANGE,
)

imageIterator = datagen.flow(np.array(images), batch_size=1)

# append augmented images to the list
for i in range(SAMPLE_BOOST_SIZE):
    images = np.append(images, imageIterator.next(), axis=0)

print("Number of images boosted: ", len(images))
# %%
# show last 5 generated images

for i in range(1, 5):
    plt.imshow(images[-i].astype(np.uint8), cmap="gray")
    plt.show()

# %%

# create normalize dataset
training_set = images / 127.5 - 1

# %%

# generator model

generator = tf.keras.Sequential()

# first dense layer
generator.add(tf.keras.layers.Dense(4 * 4 * 16, use_bias=False, input_shape=(NOISE_VECTOR_SIZE,)))
generator.add(tf.keras.layers.BatchNormalization())
generator.add(tf.keras.layers.LeakyReLU())

# reshape the output of the first dense layer
generator.add(tf.keras.layers.Reshape((4, 4, 16)))

# deconvolution layers
generator.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=1, strides=1, padding="same", use_bias=False))
generator.add(tf.keras.layers.BatchNormalization())
generator.add(tf.keras.layers.LeakyReLU())

generator.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=1, strides=1, padding="same", use_bias=False))
generator.add(tf.keras.layers.BatchNormalization())
generator.add(tf.keras.layers.LeakyReLU())


generator.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=1, strides=1, padding="same", use_bias=False))
generator.add(tf.keras.layers.BatchNormalization())
generator.add(tf.keras.layers.LeakyReLU())

generator.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=1, strides=1, padding="same", use_bias=False))
generator.add(tf.keras.layers.BatchNormalization())
generator.add(tf.keras.layers.LeakyReLU())


# output layer
generator.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=1, strides=1, padding="same", use_bias=False, activation="tanh"))


# %%

# test generator model

noise = tf.random.normal([1, NOISE_VECTOR_SIZE])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap="gray")


# %%

# discriminator model

discriminator = tf.keras.Sequential()

# convolution layers
discriminator.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=1, strides=2, padding="same", use_bias=False))
discriminator.add(tf.keras.layers.BatchNormalization())
discriminator.add(tf.keras.layers.LeakyReLU())

discriminator.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=1, strides=1, padding="same", use_bias=False))
discriminator.add(tf.keras.layers.BatchNormalization())
discriminator.add(tf.keras.layers.LeakyReLU())


discriminator.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=1, strides=1, padding="same", use_bias=False))
discriminator.add(tf.keras.layers.BatchNormalization())
discriminator.add(tf.keras.layers.LeakyReLU())

discriminator.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=1, strides=2, padding="same", use_bias=False))
discriminator.add(tf.keras.layers.BatchNormalization())
discriminator.add(tf.keras.layers.LeakyReLU())

# flatten the output of the convolution layers
discriminator.add(tf.keras.layers.Flatten())

# dense layer
discriminator.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# %%

# test discriminator model

decision = discriminator(generated_image)

print(decision)

# %%

def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMG_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMG_SIZE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)
    
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0

    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMG_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMG_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMG_SIZE, c:c + IMG_SIZE] = generated_images[image_count] * 255
            image_count += 1
    
    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filename = os.path.join(output_path, f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
# %%

optimizer = tf.keras.optimizers.Adam(LR_RATE,0.5)

discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

random_input = tf.keras.layers.Input(shape=(NOISE_VECTOR_SIZE,))

generated_image = generator(random_input)

discriminator.trainable = False

validity = discriminator(generated_image)

combined = tf.keras.Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))

fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_VECTOR_SIZE))

# %%
cnt= 1
for epoch in range(EPOCHS):
    idx = np.random.randint(0, training_set.shape[0], BATCH_SIZE)
    x_real = training_set[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_VECTOR_SIZE))
    x_fake = generator.predict(noise)

    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)

    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

    generator_metric = combined.train_on_batch(noise, y_real)

    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_noise)
        cnt += 1

        print(f"{epoch} epoch, Discriminator accuracy: {100*discriminator_metric[1]}, Generator accuracy: {100*generator_metric[1]}")
    



# %%
