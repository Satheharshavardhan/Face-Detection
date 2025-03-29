# %%
import os
import json
import cv2
import numpy as np
import albumentations as alb
import tensorflow as tf
import pickle

# %%
augmentor = alb.Compose(
    [
        alb.RandomCrop(width=450, height=450),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.RandomGamma(p=0.2),
        alb.RGBShift(p=0.2),
        alb.VerticalFlip(p=0.5),
    ],
    bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels'])
)

# %%
for partition in ['train', 'test', 'valid']:
    for image in os.listdir(os.path.join('split_data', partition, 'images')):
        img = cv2.imread(os.path.join(
            'split_data', partition, 'images', image))

        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join(
            'split_data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))

        try:
            for x in range(120):
                augmented = augmentor(
                    image=img, bboxes=[coords], class_labels=['Face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images',
                            f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)
        except Exception as e:
            print(e)

# %%
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

# %%
dataset_dict = {}

for name in ['train', 'test', 'valid']:
    dataset = tf.data.Dataset.list_files(os.path.join('aug_data', name, 'images', '*.jpg'), shuffle=False)
    dataset = dataset.map(load_image)
    dataset = dataset.map(lambda img: tf.image.resize(img, (120, 120)))
    dataset = dataset.map(lambda img: img / 255.0)
    dataset_dict[name + '_images'] = dataset

# %%
train_images = dataset_dict['train_images']
test_images = dataset_dict['test_images']
valid_images = dataset_dict['valid_images']

# %% [markdown]
# File-6

# %%
# import json
# import numpy as np
# import tensorflow as tf

# %%
# def load_labels(label_path):
#     with open(label_path.numpy(), 'r', encoding='utf-8') as f:
#         label = json.load(f)

#     return [label['class']], label['bbox']

# %%
# def set_shapes(class_label, bbox):
#     class_label.set_shape([1])
#     bbox.set_shape([4])
#     class_label = tf.cast(class_label, tf.float32)
#     return class_label, bbox

# %%
# train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
# train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
# train_labels = train_labels.map(set_shapes)

# %%
# test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
# test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
# test_labels = test_labels.map(set_shapes)

# %%
# valid_labels = tf.data.Dataset.list_files('aug_data\\valid\\labels\\*.json', shuffle=False)
# valid_labels = valid_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
# valid_labels = valid_labels.map(set_shapes)

# %% [markdown]
# File-7

# %%
# # from file_5 import train_images, test_images, valid_images
# # from file_6 import train_labels, test_labels, valid_labels
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# %%
# len(train_images), len(test_labels)

# %%
# len(test_images), len(test_labels)

# %%
# len(valid_images), len(valid_labels)

# %%
# type(train_images)

# %%
# train = tf.data.Dataset.zip((train_images, train_labels))
# train = train.shuffle(7000)
# train = train.batch(8)
# train = train.prefetch(3)

# %%
# test = tf.data.Dataset.zip((test_images, test_labels))
# test = test.shuffle(2000)
# test = test.batch(8)
# test = test.prefetch(3)

# %%
# valid = tf.data.Dataset.zip((valid_images, valid_labels))
# valid = valid.shuffle(2000)
# valid = valid.batch(8)
# valid = valid.prefetch(3)

# %%
# data_samples = train.as_numpy_iterator()

# %%
# res = data_samples.next()

# %%
# res[0]

# %%
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

# for idx in range(4):
#     sample_image = res[0][idx].copy()
#     sample_coords = res[1][1][idx]

#     start_point = tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int))
#     end_point = tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int))

#     cv2.rectangle(sample_image, start_point, end_point, (255, 0, 0), 2)
#     ax[idx].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

# plt.show()

# %% [markdown]
# File-8

# %%
# from tensorflow.keras.models import Model # type: ignore
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D # type: ignore
# from tensorflow.keras.applications import VGG16 # type: ignore
# # from file_7 import train

# %%
# vgg = VGG16(include_top=False)

# %%
# vgg.summary()

# %%
# def build_model():
#     input_layer = Input(shape=(120,120,3))

#     vgg = VGG16(include_top=False)(input_layer)

#     f1 = GlobalAveragePooling2D()(vgg)
#     class1 = Dense(2048, activation='relu')(f1)
#     class2 = Dense(1, activation='sigmoid')(class1)

#     f2 = GlobalAveragePooling2D()(vgg)
#     regress1 = Dense(2048, activation='relu')(f2)
#     regress2 = Dense(4, activation='sigmoid')(regress1)

#     facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
#     return facetracker

# %%
# facetracker = build_model()

# %%
# facetracker.summary()

# %%
# X, y = train.as_numpy_iterator().next()

# %%
# X.shape 

# %%
# classes, coords = facetracker.predict(X)

# %%
# classes, coords

# %%
# y

# %% [markdown]
# File-9

# %%
# # from file_8 import facetracker, X, y, classes, coords
# # from file_7 import train
# import tensorflow as tf

# %%
# batches_per_epoch = len(train)

# %%
# batches_per_epoch

# %%
# lr_decay = (1./0.75 - 1)/batches_per_epoch

# %%
# lr_decay

# %%
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

# %%
# def localization_loss(y_true, y_pred):
#     delta_cord = tf.reduce_sum(tf.square(y_true[:,:2]) - y_pred[:, :2])

#     h_true = y_true[:,3] - y_true[:, 1]
#     w_true = y_true[:,2] - y_true[:, 0]

#     h_pred = y_pred[:, 3] - y_pred[:, 1]
#     w_pred = y_pred[:, 2] - y_pred[:, 0]

#     delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
#     return delta_cord + delta_size

# %%
# def localization_loss(y_true, y_pred):
#     delta_cord = tf.reduce_sum(tf.square(y_true - y_pred))

#     w_true = y_true[:, 2] - y_true[:, 0]
#     h_true = y_true[:, 3] - y_true[:, 1]
#     w_pred = y_pred[:, 2] - y_pred[:, 0]
#     h_pred = y_pred[:, 3] - y_pred[:, 1]

#     delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

#     delta_cord = tf.cast(delta_cord, tf.float32)
#     delta_size = tf.cast(delta_size, tf.float32)

#     return delta_cord + delta_size

# %%
# classloss = tf.keras.losses.BinaryCrossentropy()
# regressloss = localization_loss

# %%
# localization_loss(y[1], coords).numpy()

# %%
# classloss(y[0], classes).numpy()

# %%
# classes.shape

# %%
# y

# %% [markdown]
# File-10

# %%
# from tensorflow.keras.models import Model  # type: ignore
# import tensorflow as tf
# # from file_8 import facetracker
# # from file_9 import classloss, regressloss, optimizer
# # from file_7 import train, test, valid

# %%
# class FaceTracker(Model):
#     def __init__(self, facetracker, **kwargs):
#         super().__init__(**kwargs)
#         self.model = facetracker

#     def compile(self, opt, classloss, localizationloss, **kwargs):
#         super().compile(**kwargs)
#         self.closs = classloss
#         self.lloss = localizationloss
#         self.opt = opt

#     def train_step(self, batch, **kwargs):

#         X, y = batch

#         with tf.GradientTape() as tape:
#             classes, coords = self.model(X, training=True)
#             print(f"y[0] shape: {y[0].shape}, classes shape: {classes.shape}")
#             batch_classloss = self.closs(y[0], classes)
#             batch_localizationloss = self.lloss(
#                 tf.cast(y[1], tf.float32), coords)

#             total_loss = batch_localizationloss+0.5*batch_classloss

#             grad = tape.gradient(total_loss, self.model.trainable_variables)

#         self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

#         return {"total loss": total_loss, "class loss": batch_classloss, "regression loss": batch_localizationloss}

#     def test_step(self, batch, **kwargs):
#         X, y = batch

#         classes, coords = self.model(X, training=False)

#         batch_classloss = self.closs(y[0], classes)
#         batch_regressloss = self.lloss(tf.cast(y[1], tf.float32), coords)
#         total_loss = batch_regressloss + 0.5*batch_classloss

#         return {"total loss": total_loss, "class loss": batch_classloss, "regression loss": batch_regressloss}

#     def call(self, X, **kwargs):
#         return self.model(X, **kwargs)

# %%
# model = FaceTracker(facetracker)

# %%
# model.summary

# %%
# model.compile(optimizer, classloss, regressloss)

# %%
# logdir = 'logs'

# %%
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# %%
# hist = model.fit(train, epochs=49, validation_data=valid,
#                  callbacks=[tensorboard_callback])

# %%



