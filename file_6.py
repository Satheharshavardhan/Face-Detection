# %%
import json
import numpy as np
import tensorflow as tf

# %%
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)

    return [label['class']], label['bbox']

# %%
def set_shapes(class_label, bbox):
    class_label.set_shape([1])
    bbox.set_shape([4])
    class_label = tf.cast(class_label, tf.float32)
    return class_label, bbox

# %%
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
train_labels = train_labels.map(set_shapes)

# %%
test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
test_labels = test_labels.map(set_shapes)

# %%
valid_labels = tf.data.Dataset.list_files('aug_data\\valid\\labels\\*.json', shuffle=False)
valid_labels = valid_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
valid_labels = valid_labels.map(set_shapes)

# %%



