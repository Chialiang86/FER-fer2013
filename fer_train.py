import numpy as np 
import pandas as pd #CSV file I/O
import cv2

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras import optimizers
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('input/fer2020.csv')

emotion_map = {0: 'None', 1: 'Angry', 2: 'Happy', 3: 'Sad', 4: 'Surprise'}
learningRate = 0.01
num_classes = 5
width, height = 48, 48
num_epochs = 100
batch_size = 64
num_features = 64

# git 
# 0 neu, 1 6 happy, 2 7 sad, 3 8 surprise, 4 angry
# 0 neu, 1 angry, 2 happy, 3 sad, 4 surprise

def row2image(row):
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), emotion])

#split data into training, validation and test set
data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()
print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(data_train.shape, data_val.shape, data_test.shape))
emotion_labels = ['None','Angry', 'Happy', 'Sad', 'Surprise']

# angry happy sad surprise  

a = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
def CRNO(df, dataName):
    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1,width, height,1)/255.0   
    data_Y = to_categorical(df['emotion'], num_classes) 
    print(data_Y[0]) 
    print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
    return data_X, data_Y

train_X, train_Y = CRNO(data_train, "train") #training data
val_X, val_Y     = CRNO(data_val, "val") #validation data
test_X, test_Y   = CRNO(data_test, "test") #test data

def VGG16():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', input_shape=(width, height, 1), data_format='channels_last'))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    '''
    model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    '''
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    sgd = optimizers.SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=sgd, 
                  metrics=['accuracy'])
    return model

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

#es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)

#model = Sequential()
#model = model.load_weights('2013/weights-50-0.78.h5', by_name = True)

model = VGG16()
model.summary()

#train area
filepath="weights-{epoch:02d}-{accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(data_generator.flow(train_X, train_Y, batch_size),
                                steps_per_epoch=len(train_X) / batch_size,
                                epochs=num_epochs,
                                verbose=1, 
                                callbacks = [checkpoint],
                                validation_data=(val_X, val_Y))

model.evaluate(test_X, test_Y, verbose=0)

fig, axes = plt.subplots(1,2, figsize=(18, 6))
# Plot training & validation accuracy values
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')
plt.savefig('LossAndAcc2020.png')