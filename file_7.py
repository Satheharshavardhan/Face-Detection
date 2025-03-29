# %%
from file_5 import train_images, test_images, valid_images
from file_6 import train_labels, test_labels, valid_labels
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

# %%
len(train_images), len(test_labels)

# %%
len(test_images), len(test_labels)

# %%
len(valid_images), len(valid_labels)

# %%
type(train_images)

# %%
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(7000)
train = train.batch(8)
train = train.prefetch(3)

# %%
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(2000)
test = test.batch(8)
test = test.prefetch(3)

# %%
valid = tf.data.Dataset.zip((valid_images, valid_labels))
valid = valid.shuffle(2000)
valid = valid.batch(8)
valid = valid.prefetch(3)

# %%
data_samples = train.as_numpy_iterator()

# %%
res = data_samples.next()

# %%
res[0]

# %%
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

for idx in range(4):
    sample_image = res[0][idx].copy()
    sample_coords = res[1][1][idx]

    start_point = tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int))
    end_point = tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int))

    cv2.rectangle(sample_image, start_point, end_point, (255, 0, 0), 2)
    ax[idx].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

plt.show()

# %%



