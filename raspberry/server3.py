import cv2 # openCV 4.5.1
import numpy as np # numpy 배열
import os # 파일 및 폴더의 경로 지정을 위한 모듈
import tensorflow as tf # 텐서플로우
from tensorflow import keras # 케라스
import time #프로세스 소요시간 표시 목적

from skimage.io import imread #이미지 보이기
from skimage.transform import resize # 이미지 리사이즈

from PIL import Image, ImageFont, ImageDraw #자막 투입 목적
from io import BytesIO

from collections import deque #비디오 영상에 텍스트 씌워 저장하기에 사용

base_model=keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3),
                                                  include_top=False,
                                                  weights='imagenet', classes=2)

model=keras.models.load_model('210508_vc_MobileNet_model_epoch100.h5')

input_path=0 # 노트북 웹캠을 비디오 인풋으로 설정함
output_path='output_webcam_test.mp4' #저장할 결과 동영상 파일 이름

# . ----- 비디오 스트리밍을 불러오기 & 초기설정 -----
vc = cv2.VideoCapture(input_path)
fps = vc.get(cv2.CAP_PROP_FPS)  # input_path의 초당 프레임 수 인식. fps=30.0
print(f'fps : {fps}')

writer = None
(W, H) = (None, None)
i = 0  # 비디오 초 번호. While loop를 돌아가는 회차
Q = deque(maxlen=128)

video_frm_ar = np.zeros((1, int(fps), 160, 160, 3), dtype=np.float)  # frames
frame_counter = 0  # 1초당 프레임 번호. 1~30
frame_list = []
preds = None
maxprob = None

# . While loop : 스트리밍이 끝날 때까지 프레임 추출 반복문 시작
# ----- 스트리밍 동영상들을 (30, 160, 160, 3)으로 저장하기 시작 -----
while True:
    frame_counter += 1
    grabbed, frm = vc.read()  # 비디오를 1개 프레임씩 읽는다.
    # > grabbed=True, frm=프레임별 넘파이 배열. (240, 320, 3)

    if not grabbed:  # 프레임이 안 잡힐 경우
        print('프레임이 없습니다. 스트리밍을 종료합니다.')
        break

    if fps != 30:  # 비디오 fps가 30이 아니면 루프를 돌지 않기로 한다.
        print('비디오의 초당 프레임이 30이 아닙니다. fps=30으로 맞춰주세요.')
        break

    if W is None or H is None:  # 프레임 이미지 폭(W), 높이(H)를 동영상에서
        (H, W) = frm.shape[:2]

    output = frm.copy()  # 비디오 프레임을 그대로 복사. 저장/출력할 .mp4 파일로

    frame = resize(frm, (160, 160, 3))  # > input 배열을 (160, 160, 3)으로 변환
    frame_list.append(frame)  # 각 프레임 배열 (160, 160, 3)이 append 된다.

    if frame_counter >= fps:  # 프레임 카운터가 30이 된 순간. len(frame_list)==30이 된 순간.
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

        # MobileNet로 초당 프레임 이미지 배열로부터 특성 추출 : (1*30, 5, 5, 1024)
        pred_imgarr = base_model.predict(frame_ar)  # > (30, 5, 5, 1024)
        # 추출된 특성 배열들을 1차원으로 변환 : (1, 30, 5*5*1024)
        pred_imgarr_dim = pred_imgarr.reshape(1, pred_imgarr.shape[0], 5 * 5 * 1024)  # > (1, 30, 25600)
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
    # font1 = ImageFont.truetype('fonts/Raleway-ExtraBold.ttf', 24)
    # font2 = ImageFont.truetype('fonts/Raleway-ExtraBold.ttf', 48)
    #
    # if preds is not None and maxprob is not None:  # 예측값이 발생한 후부터
    #     if (preds[0][1]) < th:  # > 폭력일 확률이 th보다 작으면 정상
    #         text1_1 = 'Normal'
    #         text1_2 = '{:.2f}%'.format(100 - (maxprob * 100))
    #         img_pil = Image.fromarray(output)
    #         draw = ImageDraw.Draw(img_pil)
    #         draw.text((int(0.025 * W), int(0.025 * H)), text1_1, font=font1, fill=(0, 255, 0, 0))
    #         draw.text((int(0.025 * W), int(0.095 * H)), text1_2, font=font2, fill=(0, 255, 0, 0))
    #         output = np.array(img_pil)
    #         # cv2.putText(이미지파일, 출력문자, 시작위치좌표(좌측하단), 폰트, 폰트크기, 폰트색상, 폰트두께)
    #
    #     else:  # > 폭력일 확률이 th보다 크면 폭력 취급
    #         text2_1 = 'Violence Alert!'
    #         text2_2 = '{:.2f}%'.format(maxprob * 100)
    #         img_pil = Image.fromarray(output)
    #         draw = ImageDraw.Draw(img_pil)
    #         draw.text((int(0.025 * W), int(0.025 * H)), text2_1, font=font1, fill=(0, 0, 255, 0))
    #         draw.text((int(0.025 * W), int(0.095 * H)), text2_2, font=font2, fill=(0, 0, 255, 0))
    #         output = np.array(img_pil)
    #
    #         # 자막이 씌워진 동영상을 writer로 저장함
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

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
cv2.destroyAllWindows()