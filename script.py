import cv2 as cv
import tensorflow as tf
from numba import cuda

device = cuda.get_current_device()
device.reset()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# tf.compat.v1.keras.backend.set_session(session)

model = tf.keras.models.load_model('CNN_64X3.h5')


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def prepare(filepath):
    IMG_SIZE = 65  # 50 in txt-based
    img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
prediction = model.predict([prepare('test.jpg')])

print(prediction[0][0])
# capture = cv.VideoCapture(0)

# while True:
#     isTrue, frame = capture.read()

#     frame_resized = rescaleFrame(frame, scale=.2)

#     #cv.imshow('Video', frame)
#     cv.imshow('Video Resized', frame_resized)

#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()
