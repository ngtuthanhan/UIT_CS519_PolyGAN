from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
import numpy as np
import os
from PIL import Image


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)
def calculate_inception_score(images, n_split=10, eps=1E-16):
    model = InceptionV3()
    scores=list()
    n_part=floor(images.shape[0]/n_split)
    for i in range(n_split):
        print(i)
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        subset = subset.astype('float32')
        subset = scale_images(subset, (299, 299, 3))
        subset = preprocess_input(subset)
        p_yx = model.predict(subset)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (log(p_yx +eps) - log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = mean(sum_kl_d)
        is_score = exp(avg_kl_d)
        scores.append(is_score)
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

images = []
images_dir= './results/test/Stage3temp_res/'
for fname in os.listdir(images_dir):
    fpath = os.path.join(images_dir, fname)
    im = Image.open(fpath)
    images.append(im)
images = np.stack(images)
# (images, _), (_, _) = cifar10.load_data()
# shuffle(images)
print('loaded', images.shape, type(images))
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)