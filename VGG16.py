import keras
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np 
import cv2

size = (32, 32, 3)
class VGG():
    def __init__(self):
        self.__num_classes = 10
        self.__epochs = 20
        self.__batchSize = 200
        self.__learningRate = 0.01
        self.__optimizer = 'SGD'

        self.__model = Sequential()
        self.__trainedModel = keras.models.load_model('maxAcc.h5')
        #self.__model = VGG16(weights='imagenet', include_top=False)
        (self.__trainImage, self.__trainLabel), (self.__testImage, self.__testLabel) = cifar10.load_data()
        self.__valImage = []
        self.__valLabel = []
        self.__history = 0
        self.__classText = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    def loadData(self):
        (self.__trainImage, self.__trainLabel), (self.__testImage, self.__testLabel) = cifar10.load_data()
    
    def setOneHotEncode(self):
        self.__trainLabel = keras.utils.to_categorical(self.__trainLabel, self.__num_classes)
        self.__testLabel = keras.utils.to_categorical(self.__testLabel, self.__num_classes)
        self.__valImage = self.__trainImage[-5000:]
        self.__valLabel = self.__trainLabel[-5000:]
        self.__trainImage = self.__trainImage[:-5000]
        self.__trainLabel = self.__trainLabel[:-5000]

    def buildModel(self):
        self.__model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', input_shape=(32,32,3)))
        self.__model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Flatten())
        self.__model.add(Dense(4096, activation = 'relu'))
        self.__model.add(Dense(4096, activation = 'relu'))
        self.__model.add(Dense(10, activation = 'softmax'))
        sgd = optimizers.SGD(lr=self.__learningRate, decay=1e-6, momentum=0.9, nesterov=True)
        self.__model.compile(loss='categorical_crossentropy', 
                  optimizer=sgd, 
                  metrics=['accuracy'])
        return self.__model
    
    def train(self):
        filepath="weights-{epoch:02d}-{accuracy:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.__history = self.__model.fit(self.__trainImage, self.__trainLabel, batch_size=self.__batchSize, \
            nb_epoch=self.__epochs, callbacks=[checkpoint], validation_data=(self.__valImage, self.__valLabel))
        return self.__history

    def printEvaluate(self):
        return self.__model.evaluate(self.__testImage, self.__testLabel, verbose=0)
    
    def showTrainImg(self):
        #32*32*3
        img10 = [0] * 10
        boolMap = [0] * 10
        for i in range(len(self.__trainLabel)):
            if boolMap[self.__trainLabel[i][0]] == 0:
                label = self.__trainLabel[i][0]
                boolMap[label] = 1
                img10[label] = self.__trainImage[i]
                if 0 not in boolMap:
                    break
        fig, ax = plt.subplots(2, 5, figsize=(8, 6))
        for i in range(10):
            ax[i//5, i%5].imshow(img10[i])
            ax[i//5, i%5].set_title(self.__classText[i])
        fig.tight_layout()
        plt.show()
    
    def printHyperparameter(self):
        hp = self.__model.get_config()
        print('hyperparameters:')
        print('batch size:', self.__batchSize)
        print('learning rate:', self.__learningRate)
        print('optimizer:', self.__optimizer)

    def printSummary(self):
        self.__model.summary()
    
    def getAccLossPlot(self):
        acc = cv2.imread('acc.png')
        loss = cv2.imread('loss.png')
        al = np.vstack((acc, loss))
        al = cv2.resize(al, (480, 720))
        cv2.imshow('Acc & Loss', al)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showTestRes(self, index):
        assert index >= 0 and index < self.__testImage.shape[0]
        img = np.reshape(self.__testImage[index], (1, 32, 32, 3))
        score = self.__trainedModel.predict(img, verbose=1)

        fig = plt.figure(figsize=(7,4))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 2])
        ax0 = fig.add_subplot(spec[0])
        ax0.imshow(np.reshape(img, (32, 32, 3)))
        ax1 = fig.add_subplot(spec[1])
        ax1.bar(self.__classText, score[0], bottom=None, align='center')
        plt.sca(ax1)
        plt.xticks(rotation='vertical')
        fig.tight_layout()
        plt.show()
   
        


model = VGG()
model.buildModel()
model.setOneHotEncode()

history = model.train()

res = model.printEvaluate()
print('Test loss:', res[0])
print('Test accuracy:', res[1])

h = history.history
print(h.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

print('Done!!!')

