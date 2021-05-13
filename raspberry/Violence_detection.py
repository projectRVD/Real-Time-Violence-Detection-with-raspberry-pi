# import
from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from collections import deque
import argparse
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO
import time

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def souhaiel_model(tf,wgts='fightw.hdfs'):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    optimizers = tf.keras.optimizers
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()

    #cnn.add(base_model)
    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
    # Freeze the layers except the last 4 layers (we will only use the base model to extract features)
    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()
    model.add(layers.TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30, return_sequences= True))
    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="sigmoid"))
    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wgts)
    rms = optimizers.RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model

model1 = souhaiel_model(tf)
print(model1.summary())

from tensorflow.keras.utils import plot_model
model1= souhaiel_model(tf)

np.random.seed(1234)
model1 = souhaiel_model(tf)

graph = tf.compat.v1.get_default_graph
graph


def video_reader(cv2,filename):
    frames = np.zeros((30, 160, 160, 3), dtype=np.float)
    i=0
    print(frames.shape)
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(160,160,3))
    frm = np.expand_dims(frm,axis=0)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    print("reading video")
    while i < 30:
        rval, frame = vc.read()
        frm = resize(frame,(160,160,3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
    return frames

def pred_fight(model,video,acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >=acuracy:
        return True , pred_test[0][1]
    else:
        return False , pred_test[0][1]

def main_fight(vidoss):
    vid = video_reader(cv2,vidoss)
    datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
    datav[0][:][:] = vid
    millis = int(round(time.time() * 1000))
    print(millis)
    f , precent = pred_fight(model1,datav,acuracy=0.65)
    millis2 = int(round(time.time() * 1000))
    print(millis2)
    res_fight = {'violence':f ,'violenceestimation':str(precent)}
    res_fight['processing_time'] =  str(millis2-millis)
    return res_fight


from collections import deque
import argparse
from skimage.transform import resize
from easydict import EasyDict
args=EasyDict()
args.size=128
print(args)


# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = model1
# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args['size'])


# initialize the video stream, pointer to output video file, andframe dimensions

####### vs=cv2.VideoCapture(args["input"])
vs=cv2.VideoCapture('C:\lab\pythonProject\KakaoTalk_20210426_104932603.mp4')
#vc=cv2.VideoCapture('Fighting013_x264.mp4')
fps = vs.get(cv2.CAP_PROP_FPS)
writer = None
(W, H) = (None, None)
#client = Client("ACea4cecca40ebb1bf4594098d5cef4541", "32789639585561088d5937514694e115") #update from twilio
prelabel = ''
ok = 'Normal'
okk='violence'
i=0
frames = np.zeros((30, 160, 160, 3), dtype=np.float)
datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
frame_counter=0

while True:
    # read the next frame from the file
    (grabbed, frm) = vs.read()
    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frm.shape[:2]
    #framecount = framecount+1
    # clone the output frame, then convert it from BGR to RGB ordering, resize the frame to a fixed 224x224,
    # and then perform mean subtraction
    output=frm.copy()
    while i < 30:
        rval, frame = vs.read()
        frame_counter +=1
        if frame_counter == vs.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = resize(frame,(160,160,3))
        frame = np.expand_dims(frame,axis=0)
        if(np.max(frame)>1):
            frame = frame/255.0
        frames[i][:] = frame
        i +=1

    datav[0][:][:] =frames
    frames -= mean

	# make predictions on the frame and then update the predictions
	# queue
    preds = model1.predict(datav)
#	print('Preds = :', preds)

#	total = (preds[0]+ preds[1]+preds[2] + preds[3]+ preds[4]+preds[5])
#	maximum = max(preds)
#	rest = total - maximum

#	diff = (.8*maximum) - (.1*rest)
#	print('Difference of prob ', diff)
#	th = 100
#	if diff > .60:
#		th = diff
#	print('Old threshold = ', th)
    prediction = preds.argmax(axis=0)
    Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
    results = np.array(Q).mean(axis=0)
    print('Results = ', results)
    maxprob = np.max(results)
    print('Maximun Probability = ', maxprob)
    i = np.argmax(results)
    rest = 1 - maxprob

    diff = (maxprob) - (rest)
    print('Difference of prob ', diff)
    th = 100
    if diff > .80:
        th = diff

    if (preds[0][1]) < th:
        text = "Alert : {} - {:.2f}%".format((ok), 100 - (maxprob * 100))
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_TRIPLEX , 1.25, (0, 255, 0), 5)
    else:
        text = "Alert : {} - {:.2f}%".format((okk), maxprob * 100)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_TRIPLEX , 1.25, (0, 0, 255), 5)
#		if label != prelabel:
#			client.messages.create(to="<+country code>< receiver mobile number>", #for example +918255555555
#                       from_="+180840084XX", #sender number can be coped from twilo
#                       body='\n'+ str(text) +'\n Satellite: ' + str(camid) + '\n Orbit: ' + location)

	# check if the video writer is None
    if writer is None:
	    # initialize our video writer
        fourcc=cv2.VideoWriter_fourcc(*"FMP4")
        writer=cv2.VideoWriter('C:\lab\pythonProject\KakaoTalk_20210426_104932603_output.mp4',
                               fourcc, 27.0, (W, H), True)

	# write the output frame to disk
    writer.write(output)

	# show the output image
    cv2.imshow("This is Output", output)
    cv2.waitKey(0)

#print('Frame count', framecount)
# release the file pointers
print("[INFO] cleaning up...")

# 작업 완료 후 해제
writer.release()
vs.release()
#vc.release()
cv2.destroyAllWindows()