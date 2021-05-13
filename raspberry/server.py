# Import
from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import imagezmq
import os
from PIL import Image
from io import BytesIO
import time
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import  Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNN + LSTM 모델 생성 함수(souhaiel_model) 선언
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

# CNN + LSTM 모델 생성(model1)
model1 = souhaiel_model(tf)
print(model1.summary())


writer = None
(W, H) = (None, None)
i = 0  # 비디오 초 번호. While loop를 돌아가는 회차
Q = deque(maxlen=128)

video_frm_ar = np.zeros((1, int(fps), 160, 160, 3), dtype=np.float)  # frames
frame_counter = 0  # 1초당 프레임 번호. 1~30
frame_list = []
preds = None
maxprob = None

# client(라즈베이 파이)에서 영상 데이터 수신
image_hub = imagezmq.ImageHub()
while True:  # show streamed images until Ctrl-C
    frame_counter += 1
    rpi_name, frm = image_hub.recv_image()
    cv2.imshow(rpi_name, frm) # 1 window for each RPi
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')

    if W is None or H is None:  # 프레임 이미지 폭(W), 높이(H)를 동영상에서
        (H, W) = frm.shape[:2]

    output = frm.copy()  # 비디오 프레임을 그대로 복사. 저장/출력할 .mp4 파일로

    frame = resize(frm, (160, 160, 3))  # > input 배열을 (160, 160, 3)으로 변환
    frame_list.append(frame)  # 각 프레임 배열 (160, 160, 3)이 append 된다.

    if frame_counter == 30:  # 프레임 카운터가 30이 된 순간. len(frame_list)==30이 된 순간.
        # . ----- 1초=30프레임마다 묶어서 예측(.predict) -----
        # . ----- 1초 동안 (1, 30, 160, 160, 3) 배열을 만들어 모델에 투입 ---
        # . ----- 예측 결과(1초)를 output에 씌워 준다. -----
        # 그러기 위해서는 30개씩 append한 리스트를 넘파이화 -> 예측 -> 리스트 초기화 과정이 필요
        frame_ar = np.array(frame_list, dtype=np.float16)  # > 30개의 원소가 든 리스트를 변환. (30, 160, 160, 3)
        frame_list = []  # 30프레임이 채워질 때마다 넘파이 배열로 변환을 마친 프레임 리스트는 초기화 해 준다.

        if (np.max(frame_ar) > 1):  # 넘파이 배열의 RGB 값을 스케일링
            frame_ar = frame_ar / 255.0

        # video_frm_ar[i][:]=frame_ar #> (i, fps, 160, 160, 3). i번째 1초짜리 영상(30프레임) 배열 파일이 된다.
        # print(video_frm_ar.shape)

        # VGG19로 초당 프레임 이미지 배열로부터 특성 추출 : (1*30, 5, 5, 512)
        pred_imgarr = base_model.predict(frame_ar)  # > (30, 5, 5, 512)
        # 추출된 특성 배열들을 1차원으로 변환 : (1, 30, 5*5*512)
        pred_imgarr_dim = pred_imgarr.reshape(1, pred_imgarr.shape[0], 5 * 5 * 512)  # > (1, 30, 12800)
        # 각 프레임 폭력 여부 예측값을 0에 저장
        preds = model.predict(pred_imgarr_dim)  # > (True, 0.99) : (폭력여부, 폭력확률)
        print(f'preds:{preds}')
        Q.append(preds)  # > Deque Q에 리스트처럼 예측값을 추가함

        # 지난 5초간의 폭력 확률 평균을 result로 한다.
        if i < 5:
            results = np.array(Q)[:i].mean(axis=0)
        else:
            results = np.array(Q)[(i - 5):i].mean(axis=0)
        # results=np.array(Q).mean(axis=0)
        print(f'Results = {results}')  # > ex : (0.6, 0.650)

        # 예측 결과에서 최대폭력확률값
        maxprob = np.max(results)  # > 가장 높은 값을 선택함
        print(f'Maximum Probability : {maxprob}')
        print('')

        rest = 1 - maxprob  # 폭력이 아닐 확률
        diff = maxprob - rest  # 폭력일 확률과 폭력이 아닐 확률의 차이
        th = 100

        if diff > 0.80:  # 폭력일 확률과 아닐 확률의 차이가 0.8 이상이면
            th = diff  # 근거가?

        frame_counter = 0  # > 1초(30프레임)가 경과했으므로 frame_counter=0으로 리셋
        i += 1  # > 1초 경과 의미

        # frame_counter==30이 되면 0으로 돌아가 위 루프를 반복해 준다.

    # ----- output에 씌울 자막 설정하기 -----
    # 30프레임(1초)마다 갱신된 값이 output에 씌워지게 된다
    scale = 1
    fontScale = min(W, H) / (300 / scale)

    if preds is not None and maxprob is not None:  # 예측값이 발생한 후부터
        if (preds[0][1]) < th:  # > 폭력일 확률이 th보다 작으면 정상
            text1_1 = 'Normal'
            text1_2 = '{:.2f}%'.format(100 - (maxprob * 100))
            cv2.putText(output, text1_1, (int(0.025 * W), int(0.1 * H)),
                        cv2.FONT_HERSHEY_TRIPLEX, fontScale, (0, 255, 0), 2)
            cv2.putText(output, text1_2, (int(0.025 * W), int(0.2 * H)),
                        cv2.FONT_HERSHEY_TRIPLEX, fontScale, (0, 255, 0), 2)
            # cv2.putText(이미지파일, 출력문자, 시작위치좌표(좌측하단), 폰트, 폰트크기, 폰트색상, 폰트두께)

        else:  # > 폭력일 확률이 th보다 크면 폭력 취급
            text2_1 = 'Violence Alert!'
            text2_2 = '{:.2f}%'.format(maxprob * 100)
            cv2.putText(output, text2_1, (int(0.025 * W), int(0.1 * H)),
                        cv2.FONT_HERSHEY_TRIPLEX, fontScale, (0, 0, 255), 2)
            cv2.putText(output, text2_2, (int(0.025 * W), int(0.2 * H)),
                        cv2.FONT_HERSHEY_TRIPLEX, fontScale, (0, 0, 255), 2)

            # 자막이 씌워진 동영상을 writer로 저장함
    if writer is None:
        writer = cv2.VideoWriter(output_path, -1, 30, (W, H), True)

    # 아웃풋을 새창으로 열어 보여주기
    cv2.imshow('This is output', output)
    writer.write(output)  # output_path로 output 객체를 저장함

    key = cv2.waitKey(round(1000 / fps))  # 프레임-다음 프레임 사이 간격
    if key == 27:  # esc 키를 누르면 루프로부터 벗어나고 output 파일이 저장됨
        print('ESC키를 눌렀습니다. 녹화를 종료합니다.')
        break

print('종료 처리되었습니다. 메모리를 해제합니다.')
writer.release()
vc.release()



# 비디오 스트리밍
    vs = cv2.VideoCapture()
    vs


# 이미지 프레임 30개씩 묶음 저장 후 딥러닝 모델에 input(shape = (1,30,160,160,3)) 후 예측값 출력
    i = 1
    images = []
    while i <= 30 :
        images.append(image)
        i += 1

    images = np.array(images).reshape((1, 30, 160, 160, 3))
    predict_result = model1.predict(images)
    print(predict_result)


