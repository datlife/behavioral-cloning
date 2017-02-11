import argparse
import base64
import os
import json
import shutil
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import time
import numpy as np
from scipy.misc import imresize
from keras.models import model_from_json
from FLAGS import *
from utils.car_helper import convert_buckets_to_steer_angle
from datetime import datetime

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # frame_block = []
        # for i in range(TIME_STEPS*BATCH_SIZE):
        #     img = data["image"]
        #     image = Image.open(BytesIO(base64.b64decode(img)))
        #     image = np.asarray(image)
        #     image = imresize(image, 0.5)
        #     image = image[30:70, :, :]
        #     frame_block.append(image)
        # frame_block = np.reshape(frame_block, newshape=[BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH, CHANNELS])
        # prediction = model.predict_on_batch(frame_block)[0]
        #
        # throttle = 0.3
        # steering_angle = prediction[0][-1]
        # POST STEER ANGLE PROCESSING
        # The current steering angle of the car

        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image = np.asarray(image)
        image = imresize(image, 0.5)
        steering_angle = model.predict(image[None, :, :, :], batch_size=1)[0][0]
        throttle = 0.2
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help='Path to model json file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    with open(args.model, 'r') as jfile:
       json_model = jfile.read()
       model = model_from_json(json_model)

    print("Load model successfully")

    model.compile("adam", "mse")
    weight_file = args.model.replace('json', 'h5')
    model.load_weights(weight_file)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    print("Model initialized...")
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

