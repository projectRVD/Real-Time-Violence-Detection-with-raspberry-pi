import cv2
import numpy as np
import imagezmq

image_hub = imagezmq.ImageHub()
while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    cv2.imshow(rpi_name, image) # 1 window for each RPi
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')
    cap = cv2.VideoCapture()

# 열렸는지 확인
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

# 웹캠의 속성 값을 받아오기
# 정수 형태로 변환하기 위해 round
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 1프레임과 다음 프레임 사이의 간격 설정
    delay = round(1000/fps)

# 웹캠으로 찰영한 영상을 저장하기
# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    out = cv2.VideoWriter('C:/Users/leesookwang/Desktop/project/sample_video10.avi', fourcc, fps, (w, h))

# 제대로 열렸는지 확인
    if not out.isOpened():
        print('File open failed!')
        cap.release()
        sys.exit()