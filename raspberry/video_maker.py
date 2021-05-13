import cv2
import numpy as np
import imagezmq

image_hub = imagezmq.ImageHub()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:/Users/leesookwang/Desktop/project/fight06.mp4', -1, 50.0, (160,160))
while True:  # show streamed images until Ctrl-C
    cv2.waitKey(1)
    rpi_name, image = image_hub.recv_image()

    if rpi_name == 'raspberrypi':
        out.write(image)
        cv2.imshow('frame',image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else :
        break
    image_hub.send_reply(b'OK')

out.release()
cv2.destroyAllWindows()