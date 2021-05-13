# 21.05.13 CODE Updated

## Server(.py)
**Run it on your desktop**
* server2.py must be compiled with the client.py
* server3.py can be used alone if only camera devices such as laptop webcams are connected.

## In order to compile server.py normally, the following files must be additionally saved.
**Font File :** https://fonts.google.com/specimen/Raleway?query=ralewa#license
* If you want to use a different font, modify the'ImageFont.truetype' part and use it.

**Weight File(.h5) :** https://drive.google.com/u/0/uc?export=download&confirm=r7yc&id=1z4UKAkhyItFCJ7aue3FNfCVN5jfsdkkU
* The deep learning model used is MobileNet. For more information, refer to the Jupiter Directory.

## Client(.py)
**Run it on your raspberrypi**
* **Use model** : Raspberry Pi 3+

## OpenCV must be installed on the Raspberry Pi.
* This is the OpenCV(ver 4.1.2) download link I referenced.
**Download Link :** https://make.e4ds.com/make/learn_guide_view.asp?idx=116


![210512_MobileNet_model_accuracy_epoch100](https://user-images.githubusercontent.com/75024126/117928225-3690d780-b336-11eb-94c9-e22fbc7a681c.png)
![210512_MobileNet_model_loss_epoch100](https://user-images.githubusercontent.com/75024126/117928234-38f33180-b336-11eb-9427-f2ed5f119339.png)
