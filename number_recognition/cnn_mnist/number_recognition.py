import tensorflow as tf
import cv2
import numpy as np
import sys

# make background white or black (I forgot)
def gray_white(gray, thresh=50):
    binary_mask = np.zeros_like(gray)
    binary_mask[gray > thresh] = 1
    #print(gray.max(), gray.min())
    #plt.imshow(binary_mask)
    my_gray = gray.copy()
    my_gray[binary_mask == 1] = 255
    return my_gray

# processing pipeline
def process_image(img):
    img = cv2.resize(img, (28,28))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_white(gray, 40)
    gray = cv2.bitwise_not(gray)
    return gray

def number_recognition():
    # get the file direction
    file_dir = './test_images/test_1.jpg'
    if(len(sys.argv)>1):
        file_dir = sys.argv[1]

    # load image
    img = cv2.imread(file_dir)

    # perform image process
    img_proced = process_image(img)
    my_array = np.array([img_proced.reshape(-1)])

    # load model
    loaded_graph = tf.Graph()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./cnn_data.meta')
        new_saver.restore(sess, './cnn_data')
        predict_op = tf.get_collection('predict_op')[0]
        hparams = tf.get_collection("hparams")
        x = hparams[0]
        keep_prob = hparams[1]

        prediction = sess.run(predict_op,feed_dict = {x: my_array, keep_prob: 1.0})

    print(np.argmax(prediction, axis=1))

    # print(img)
    cv2.imshow('img', img)
    cv2.imshow('img_proced', img_proced)
    cv2.waitKey(0)

# main function for testing
if __name__ == "__main__":
    number_recognition()
