## OpenCV-Projects 
# it has code and APK for binary classification of Open/Closed eyes.

This work develop 3 stages in order to classify Open/Closed Eyes.

For each frame, let's say:

![](ScreeShots/org.png)

1. Face detection - LBP it is performed:

![](ScreenShots/face.png)

- Then changing ROI

![](ScreenShots/face_cropped.png)

- Then both eyes are located according to geometry of the face

![](ScreenShots/Eyes_Geometry.png)


- Now changing ROI for each eye

Left Eye
![](ScreenShots/Left_Eye.png)

Right Eye
![](ScreenShots/Right_Eye.png)

2. Eye detection - Haar it is performed (Different detector for each one), and we got:

    Left Eye
![](ScreenShots/Left_Eye_Haar.png)

    Right Eye
![](ScreenShots/Right_Eye_Haar.png)

3. Binary classification according to both eyes:

![](ScreenShots/open.png)

![](ScreenShots/closed.png)
