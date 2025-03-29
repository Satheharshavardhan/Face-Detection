# %%
from file_8 import facetracker, X, y, classes, coords
from file_7 import train
import tensorflow as tf

# %%
batches_per_epoch = len(train)

# %%
batches_per_epoch

# %%
lr_decay = (1./0.75 - 1)/batches_per_epoch

# %%
lr_decay

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

# %%
def localization_loss(y_true, y_pred):
    delta_cord = tf.reduce_sum(tf.square(y_true[:,:2]) - y_pred[:, :2])

    h_true = y_true[:,3] - y_true[:, 1]
    w_true = y_true[:,2] - y_true[:, 0]

    h_pred = y_pred[:, 3] - y_pred[:, 1]
    w_pred = y_pred[:, 2] - y_pred[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_cord + delta_size

# %%
def localization_loss(y_true, y_pred):
    delta_cord = tf.reduce_sum(tf.square(y_true - y_pred))

    w_true = y_true[:, 2] - y_true[:, 0]
    h_true = y_true[:, 3] - y_true[:, 1]
    w_pred = y_pred[:, 2] - y_pred[:, 0]
    h_pred = y_pred[:, 3] - y_pred[:, 1]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    delta_cord = tf.cast(delta_cord, tf.float32)
    delta_size = tf.cast(delta_size, tf.float32)

    return delta_cord + delta_size

# %%
classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# %%
localization_loss(y[1], coords).numpy()

# %%
classloss(y[0], classes).numpy()

# %%
classes.shape

# %%
y

# %%



