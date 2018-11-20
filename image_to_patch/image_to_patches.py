import tensorflow as tf
from skimage import io
import matplotlib.pyplot as plt


def image_to_patches(image, patch_height, patch_width):
    # resize image so that it's dimensions are dividable by patch_height and patch_width
    image_height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    image_width = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    height = tf.cast(tf.ceil(image_height / patch_height) * patch_height, dtype=tf.int32)
    width = tf.cast(tf.ceil(image_width / patch_width) * patch_width, dtype=tf.int32)

    num_rows = height // patch_height
    num_cols = width // patch_width
    # make zero-padding
    image = tf.squeeze(tf.image.resize_image_with_crop_or_pad(image, height, width))

    # get slices along the 0-th axis
    image = tf.reshape(image, [num_rows, patch_height, width, -1])
    # h/patch_h, w, patch_h, c
    image = tf.transpose(image, [0, 2, 1, 3])
    # get slices along the 1-st axis
    # h/patch_h, w/patch_w, patch_w,patch_h, c
    image = tf.reshape(image, [num_rows, num_cols, patch_width, patch_height, -1])
    # num_patches, patch_w, patch_h, c
    image = tf.reshape(image, [num_rows * num_cols, patch_width, patch_height, -1])
    # num_patches, patch_h, patch_w, c
    image = tf.transpose(image, [0, 2, 1, 3])

    ###########################
    
    # num_patches, patch_w, patch_h, c
    image = tf.transpose(image, [0, 2, 1, 3])
    
    image = tf.reshape(image, [num_rows, num_cols * patch_width, patch_height, -1])

    image = tf.transpose(image, [0, 2, 1, 3])
    
    image = tf.reshape(image, [num_rows * patch_height, width, -1])
    
    return image

image = io.imread('laska.png')
print('Original image shape:', image.shape)
tile_size = 3
image = tf.constant(image)
tiles = image_to_patches(image, tile_size, tile_size)

sess = tf.Session()
I, tiles = sess.run([image, tiles])
print(I.shape)
print(tiles.shape)

plt.imshow(tiles)
plt.show()
