from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('weights-67-0.92.h5')
emotion_map = {0: 'None', 1: 'Angry', 2: 'Happy', 3: 'Sad', 4: 'Surprise'}
expect = ['happy', 'angry', 'sad', 'normal']
im = []
# 0 neu, 1 angry, 2 happy, 3 sad, 4 surprise

for i in range(23):
    im.append(cv2.imread('test/' + str(i) + '.jpg'))
    im[i] = cv2.cvtColor(im[i], cv2.COLOR_BGR2GRAY)
    im[i] = cv2.resize(im[i], (48, 48))

im = np.reshape(im, [23, 48, 48, 1])

res = model.predict_classes(im) 

for i in range(23):
    msg = 'index: {} res: {}'.format(i, emotion_map[res[i]])
    print( msg)

    
# pip freeze