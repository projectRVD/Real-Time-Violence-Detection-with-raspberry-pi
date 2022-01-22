# Project RVD : Realtime Violence Detection with Raspberry Pi

* Client: J.Oh
* Starts at: 2021/04/08
* Ends at: 2021/05/17
* Members: [Sukwang Lee](https://github.com/SookwangLee), [Hyewon Rho](https://github.com/rhohye22), [Jeonghun Park](https://github.com/berojung), [Jimin Chae](https://github.com/regenesis90), [Yeong-Min Choi](https://github.com/kmccym)

# *Introduction*

* Violence detection by using MobileNet + LSTM (Binary classification : Violence / Non-Violence)
* Add captions of violence dection result on video screen (video file or realtime video streaming) **[>> Fast Link](https://github.com/projectRVD/Real-Time-Violence-Detection-with-raspberry-pi/tree/main/ver_jupyter)**
* Using Raspberry Pi & camera module for realtime video streaming **[>> Fast Link](https://github.com/projectRVD/Real-Time-Violence-Detection-with-raspberry-pi/tree/main/raspberry)**

# *Environment*

- Linux OS
- Python ≥ 3.8
- OpenCV ≥ 4
- Tensorflow ≥ 2.4.0
- Keras 2.4.0

# *Tools*

- Jupyterlab ≥ 3.0
- Pycharm ≥ 2
- Raspberry Pi +3
- Camera Module

# *Result*

## Examples

![ezgif-4-3dc74782d191](https://user-images.githubusercontent.com/75024126/117774684-b9e9f480-b274-11eb-978a-060f21ffd1af.gif)
![ezgif-4-15e71033eab4](https://user-images.githubusercontent.com/75024126/117774703-bce4e500-b274-11eb-8e3c-14f54d7a8743.gif)
![ezgif-4-7a9fd72fa7c9](https://user-images.githubusercontent.com/75024126/117774858-dd14a400-b274-11eb-941a-aaf8e45eb8a7.gif)
![ezgif-4-2946841c66b2](https://user-images.githubusercontent.com/75024126/117777516-a429fe80-b277-11eb-81b0-1da2b6a0ef41.gif)

![KakaoTalk_20210514_164231168](https://user-images.githubusercontent.com/76435473/118238485-8a84f300-b4d3-11eb-89f8-8bda9e57c397.gif)


## Change of Accuracy & Loss of Model

* **Result Log** : **[View Table](https://github.com/projectRVD/Real-Time-Violence-Detection-with-raspberry-pi/blob/main/RVD_result_log.csv)**

![RVD_result_model_comparison](https://user-images.githubusercontent.com/75024126/117956567-21c33c80-b354-11eb-9768-aac0ed1fc5ef.png)
![RVD_result_log](https://user-images.githubusercontent.com/75024126/117956574-238d0000-b354-11eb-81ff-de111fa69851.png)

# *Reference*

## Datasets

* RWF2000 : https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
* Hocky : http://visilab.etsii.uclm.es/personas/oscar/FightDetection/index.html
* raw.zip : https://github.com/niyazed/violence-video-classification
* cam1, cam2 : https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/master/violence-detection-dataset/non-violent/cam1
* RVD : -

## Examples
* UCF Anomaly Detection Datasets : https://webpages.uncc.edu/cchen62/dataset.html
  * Those video files were used only for making output videos & GIFs.

## We're Inspired from them

* [Pedro Frodenas's Github](https://github.com/pedrofrodenas/Violence-Detection-CNN-LSTM/blob/master/violence_detection.ipynb)
* [Souhaiel BenSalem's Github](https://github.com/shouhaiel1/CNN-LSTM-Violence-detection)
