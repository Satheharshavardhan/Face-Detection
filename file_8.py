# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from file_7 import train

# %%
vgg = VGG16(include_top=False)

# %%
vgg.summary()

# %%
def build_model():
    input_layer = Input(shape=(120,120,3))

    vgg = VGG16(include_top=False)(input_layer)

    f1 = GlobalAveragePooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    f2 = GlobalAveragePooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

# %%
facetracker = build_model()

# %%
facetracker.summary()

# %%
X, y = train.as_numpy_iterator().next()

# %%
X.shape

# %%
classes, coords = facetracker.predict(X)

# %%
classes, coords

# %%
y

# %%



