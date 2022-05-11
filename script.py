import cv2 as cv
from matplotlib.pyplot import text
import tensorflow as tf
from numba import cuda
import numpy as np
from keras.models import model_from_json
# import math

# resetting GPU
device = cuda.get_current_device()
device.reset()

# configuring GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def process_input(img):
    IMG_SIZE = 69
    # gry_img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# def detected_mask():
#     # model = tf.keras.models.load_model('CNN_64X3.h5')
#     prediction = mask_model.predict([process_input('test/test.jpg')])
#     maxindex = int(prediction[0][0])
#     print(math.ceil(prediction[0][0]))
#     print(np.argmax(prediction))
#     print(maxindex)
#     print(className[maxindex])


def main():
    while True:
        _, frame = capture.read()

        face_detector = cv.CascadeClassifier(
            'models/haarcascade_frontalface_default.xml')

        frame = rescaleFrame(frame, 0.3)

        gray_face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray_face, scaleFactor=1.09, minNeighbors=7,  minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

        # print(faces)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            rec_frame = gray_face[y:y + h, x:x + w]
            cropped_img = process_input(rec_frame)

            prediction = mask_model.predict([cropped_img])
            maxindex = round(prediction[0][0])

            str_classname = "{} {}".format(
                className[maxindex], "{:.2f}".format(prediction[0][0], 2))
            str_facecount = "face count : {}".format(len(faces))

            cv.putText(frame, str_classname, (x+5, y-20),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (15, 15, 253) if maxindex == 0 else (0, 255, 0), 2, cv.LINE_AA)

            cv.putText(frame, str_facecount,
                       (40, 40),  font, 1, (255, 0, 0), 2, cv.LINE_AA)
            #frame_resized = rescaleFrame(frame, scale=.2)

            cv.imshow('Mask-detection', frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break
        elif 0xFF == ord('q'):
            cv.waitKey(1000)

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    className = {0: "no mask", 1: "masked"}

    # Loading the model
    json_file = open('models/mask_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    mask_model = model_from_json(loaded_model_json)
    mask_model.load_weights("models/CNN_64X3.h5")

    font = cv.FONT_HERSHEY_SIMPLEX
    capture = cv.VideoCapture('test/test2.mp4')
    # for web cam
    # capture = cv.VideoCapture(0)
    main()
