import tensorflow as tf
import cv2
import numpy as np
import glob

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
    image_shape = img.shape
    img = img[5:image_shape[0]-5,5:image_shape[1]-5]
    img = cv2.resize(img, (28,28))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_white(gray, 130)
    gray = cv2.bitwise_not(gray)
    return gray

# load images by path
def load_images(image_paths):
    my_array = None
    for path in image_paths:
        test_image = cv2.imread(path)
        processed_image = process_image(test_image)
        image_array = processed_image.reshape(-1)
        if my_array == None:
            my_array = image_array
        else:
            my_array = np.vstack((my_array,image_array))
    return my_array


if __name__ == "__main__":
    # load images
    image_paths = glob.glob('test_images/*.png')
    my_array = load_images(image_paths)

    # load the model
    loaded_graph = tf.Graph()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./cnn_data.meta')
        new_saver.restore(sess, './cnn_data')
        predict_op = tf.get_collection('predict_op')[0]
        hparams = tf.get_collection("hparams")
        x = hparams[0]
        keep_prob = hparams[1]
        predicted_logits = sess.run(predict_op,feed_dict = {x: my_array, keep_prob: 1.0})

    print(np.argmax(predicted_logits, axis=1))


