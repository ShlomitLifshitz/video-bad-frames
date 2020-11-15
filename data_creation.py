import numpy as np
import os
import glob
import keras
from keras_video import VideoFrameGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
#from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model



# some global params
SIZE = (112, 112)
CHANNELS = 3
NBFRAME = 5
BS = 1

def create_video_frames_generator():
    glob_pattern='short_videos_2/*.avi'
    gen = VideoFrameGenerator(glob_pattern=glob_pattern,
                                nb_frames=NBFRAME,
                                target_shape=SIZE,
                                batch_size=BS,
                                nb_channel=CHANNELS,
                                use_frame_cache=True)
    return gen 


def remove_batch_size(gen):
    new_gen = []
    for i in range(len(gen)):
        new_gen.append(gen[i][0][0])
    return np.array(new_gen)


def create_labeled_list(y):
    return np.ones((len(y),5), dtype=int)


def create_mix_frames(x,y):
    bad_video_index = np.random.choice(range(len(x)), round(len(x)*0.1))
    good_video_index = list(set(range(len(x)))-set(bad_video_index))
    for bad_i in bad_video_index:
        bad_f = np.random.choice(5,1)
        y[bad_i, bad_f] = 0
        good_i = np.random.choice(good_video_index,1)
        good_f = np.random.choice(5,1)
        x[bad_i,bad_f, :] = x[good_i, good_f, :]


def split_train_test(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model():
    K.clear_session()
    input_layer = Input((5, 112, 112, 3))

    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    pooling_layer = MaxPool3D(pool_size=(1, 1, 1))(conv_layer2)
    pooling_layer = BatchNormalization()(pooling_layer)
    flatten_layer = Flatten()(pooling_layer)

    dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.2)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.2)(dense_layer2)
    dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)
    dense_layer3 = Dropout(0.2)(dense_layer3)
    output_layer = Dense(units=5, activation='sigmoid')(dense_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model


if __name__ == "__main__":
    gen_data = create_video_frames_generator()
    x = remove_batch_size(gen_data)
    y = create_labeled_list(x)
    create_mix_frames(x, y)
    X_train, X_test, y_train, y_test = split_train_test(x, y)
    model = create_model()
    X_train = X_train.reshape(X_train.shape[0], 5, 112, 112, 3)
    X_test = X_test.reshape(X_test.shape[0], 5, 112, 112, 3)
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
    model.fit(x=X_train, y=y_train, batch_size=128, epochs=30, validation_split=0.2)
    model.save('video_model')
    pred = model.predict(X_test)
    results = model.evaluate(X_test, y_test)
    print("test loss, test acc:", results)
    

