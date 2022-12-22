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
# Helper functions and variables

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# print progress bar
def progress(i, size, status="Progress"):
    progress = i / size * 100
    progress = round(progress, 2)
    print("\r{}: {}%".format(status, progress), end="")

# find number of images in the folder
num_img = len(os.listdir("balaban_resim_cropped"))

# %%
# Arguments

# preprocessing arguments
ART_DIR = "balaban_resim_cropped"  # directory of the images
DATA_PATH = "data"  # path to save the dataset
IMG_PATH = "train_images"  # path to save the generated images

GENERATE_RES = 1  # Generation resolution factor (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES  # rows/cols (should be square)
IMAGE_CHANNELS = 3  # number of channels of the images (RGB = 3, Grayscale = 1)

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# %%

# boosting arguments
# if sample boost is selected add these arguments
SAMPLE_BOOST_SIZE = num_img * 200 # number of samples to boost = 46 * 100 = 4600
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.25
HEIGHT_SHIFT_RANGE = 0.25
ZOOM_RANGE = 0.4
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
FILL_MODE = "reflect"
BRIGHTNESS_RANGE = [0.5, 1.5]
SHEAR_RANGE = 0.05

# %%

# noise reduction for the filters (exclusive arguments)
MEDIAN_FILTER = False

# if gausian filter is selected add this argument
GAUSIAN_FILTER = False
SIGMAX = 0

# if biliteral filter is selected add this argument
BILITERAL_FILTER = True
D = 9  # diameter of each pixel neighborhood that is used during filtering
SIGMACOLOR = 75
SIGMASPACE = 75

# %%
# model arguments
BATCH_SIZE = 32  # batch size
EPOCHS = 1000  # number of epochs
BUFFER_SIZE = 60000  # buffer size for shuffling the dataset
LR_RATE = 1.5e-4  # learning rate

# %%

# Arguments check
if MEDIAN_FILTER and GAUSIAN_FILTER or MEDIAN_FILTER and BILITERAL_FILTER or GAUSIAN_FILTER and BILITERAL_FILTER:
    raise Exception("Only one filter can be selected at a time!")


# %%
# Filtering functions

def median_filter(img):
    return cv2.medianBlur(img, ksize=3)

def gausian_filter(img):
    return cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=SIGMAX)

def biliteral_filter(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# %%
# image loading and image augmentation (sample boost)

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)

# get number of images in the folder in IMG_PATH
num_train_img = len(os.listdir(IMG_PATH))

# if img path does not exist boost the samples and save them
if num_train_img != SAMPLE_BOOST_SIZE + num_img:
    # delete the images in the folder if there are any
    for filename in os.listdir(IMG_PATH):
        os.remove(IMG_PATH + "/" + filename)

    # create a numpy list to store the images
    images = np.empty((0, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
    for filename in os.listdir(ART_DIR):
        img = tf.keras.utils.load_img(ART_DIR + "/" + filename)
        # resize the images
        resized_img = img.resize((GENERATE_SQUARE, GENERATE_SQUARE))

        # convert to numpy array
        img_array = tf.keras.utils.img_to_array(resized_img)

        # select the filter by the arguments
        if MEDIAN_FILTER:
            img_array = median_filter(img_array)
        elif GAUSIAN_FILTER:
            img_array = gausian_filter(img_array)
        elif BILITERAL_FILTER:
            img_array = biliteral_filter(img_array)

        if img is not None:
            images = np.append(images, [img_array], axis=0)
    

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        vertical_flip=VERTICAL_FLIP,
        zoom_range=ZOOM_RANGE,
        fill_mode=FILL_MODE,
        brightness_range=BRIGHTNESS_RANGE,
        shear_range=SHEAR_RANGE,
    )


    imageIterator = datagen.flow(np.array(images), batch_size=1)


    # append augmented images to the list
    for i in range(SAMPLE_BOOST_SIZE):
        aumented_img = imageIterator.next()
        images = np.append(images, aumented_img, axis=0)
        
        # print progress
        progress(i, SAMPLE_BOOST_SIZE, status="Boosting samples")
        

    print("Number of images boosted: ", len(images), end="\r")

    
    for i in range(len(images)):
        img = tf.keras.preprocessing.image.array_to_img(images[i])
        img.save(IMG_PATH + "/" + str(i) + ".jpg")


# %%

# reshape the images for 4d tensor
training_data = np.array(images).reshape(-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)
training_data = training_data.astype("float32")
training_data = training_data / 127.5 - 1  # normalize the images to [-1, 1]

training_data = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# %%

def build_generator(seed_size, channels):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(tf.keras.layers.Reshape((4,4,256)))

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(256,kernel_size=3,padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))

   
    # Output resolution, additional upsampling
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128,kernel_size=3,padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))

    # Output resolution, additional upsampling
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128,kernel_size=3,padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))

    if GENERATE_RES>1:
      model.add(tf.keras.layers.UpSampling2D(size=(GENERATE_RES,GENERATE_RES)))
      model.add(tf.keras.layers.Conv2D(128,kernel_size=3,padding="same"))
      model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
      model.add(tf.keras.layers.Activation("relu"))

    # Final CNN layer
    model.add(tf.keras.layers.Conv2D(channels,kernel_size=3,padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))

    return model


def build_discriminator(image_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, 
                     padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

# %%
# Test the generator

generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])

# %%
# Test the discriminator
image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

discriminator = build_discriminator(image_shape)
decision = discriminator(generated_image)
print (decision)

# %%
def save_images(cnt,noise):
  image_array = np.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), IMAGE_CHANNELS), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] \
            = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)

# %%

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
# %%

generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
# %%
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(\
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(\
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, generator.trainable_variables))
    
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, 
        discriminator.trainable_variables))
  return gen_loss,disc_loss
# %%
def train(dataset, epochs):
  gen_losses = []
  disc_losses = []
  fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, 
                                       SEED_SIZE))
  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
           f' {hms_string(epoch_elapsed)}')
    save_images(epoch,fixed_seed)

    # add generator loss and discriminator loss to lists
    gen_losses.append(g_loss)
    disc_losses.append(d_loss)


  elapsed = time.time()-start
  print (f'Training time: {hms_string(elapsed)}')

  return gen_losses, disc_losses

# %%

gen_losses, disc_losses = train(training_data, EPOCHS)



# %%
# plot generator loss and discriminator loss by epoch
plt.plot(gen_losses, label='Generator Loss')
plt.plot(disc_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%

# generate GIF of all images in output folder
anim_file = 'dcgan.gif'

images_data = []
for i in range(EPOCHS):
    filename = DATA_PATH+'/output/train-' + str(i) + '.png'
    images_data.append(imageio.imread(filename))
imageio.mimsave(anim_file, images_data, fps=20)

# %%
import IPython
if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)


# %%
