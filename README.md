Driving Behavioral Cloning
==========================

**Status**: Completed

This project is to mimic human driving behavior into a car. 



### Dependencies

This project requires users to have additional libraries installed in order to use. Udacity provided a good solution by sharing  a ready-to-use environment `CarND-Term1-Starter-Kit` for students. Please refer [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) on how to install.

### How to use:

1. Download CarND Udacity Simulator ([here]()]
2. Open car simulator in autonomous mode
3. Open a terminal, run `python drive.py model/cnn.h5` (cnn.h5 is my pre-trained model)
4. Enjoy your first autonomous drive!

#### Result:

| Track 1       | Track 2       | 
| ------------- |---------------|
| ![alt text](https://github.com/dat-ai/behavioral-cloning/blob/master/docs/track1.gif)      | ![alt text](https://github.com/dat-ai/behavioral-cloning/blob/master/docs/track2.gif)|

    
## 1. Deep ResNet Pre-Activation works well

#### 1.1 Network Architecture Considerations

#### 1.2 Future goal, Recurrent Neural Network + CNN

## 2. Data Augmentation is needed.

#### 2.1 OpenCV is wonderful
The goal of data augmentation is to assist the model generalize better. In this project, I re-used to image tools from project 2 which take an image and perform multiple transformations, which are blurring, rotation and chaning brightness of an image.. randomly!
```shell
def random_transform(img):
    # There are total of 3 transformation
    # I will create an boolean array of 3 elements [ 0 or 1]
    a = np.random.randint(0, 2, [1, 3]).astype('bool')[0]
    if a[1] == 1:
        img = rotate(img)
    if a[2] == 1:
        img = blur(img)
    if a[3] == 1:
        img = gamma(img)
    return img
```
#### 2.2 Flipped that image!

You might found that during training you unconsciously was biased toward one side of street. So flipping the image helps your model generalizes better as well as. As suggested by Udacity, driving in opposite direction also helps your model. The reason is the lap has too many left turns. By driving in reversed direction, you force your model to learn the right turn too. Here are my few examples that indicates your model might not generalize enough:

1. Turn too the left or right (only)
![bad-turn]()

2. Stay on one side of the street.

## 3. Know when to stop
#### 3.1 Training strategies
#### 3.2 Becareful to high learning rate

## 4. From simulator to real RC racing car

Finally, I would like to advertise my current project [Autonomous 1/10th Self-Racing Car](https://github.com/dat-ai/jetson-car) . I applied  what I learned from Behavioral Cloning into a real RC car. This a wonderful chance to validate my model in real track. My car is used NVIDIA Jetson TK1 as a brain to control the steering servo and ESC (You can used Raspberry Pi 3 but it could be slow).
