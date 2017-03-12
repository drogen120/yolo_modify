import sys
import os

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


def process_predicts(predicts):
    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]
    threshold = 0.02

    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 2, 1))

    P = C * p_classes

    filter_mat_probs = np.array(P > threshold, dtype='bool')

    filter_mat_boxes = np.nonzero(filter_mat_probs)

    #print P[5,1, 0, :]

    # index = np.argmax(P)
    #
    # index = np.unravel_index(index, P.shape)

    class_num = filter_mat_boxes[3]

    coordinate = np.reshape(coordinate, (7, 7, 2, 4))

    max_coordinate = coordinate[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2], :]

    xcenter = max_coordinate[:,0]
    ycenter = max_coordinate[:,1]
    w = max_coordinate[:,2]
    h = max_coordinate[:,3]

    xcenter = (filter_mat_boxes[1] + xcenter) * (448/7.0)
    ycenter = (filter_mat_boxes[0] + ycenter) * (448/7.0)

    w = w * 448
    h = h * 448

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0

    xmax = xmin + w
    ymax = ymin + h

    result = []

    for i in range(len(class_num)):
        result.append([xmin[i], ymin[i], xmax[i], ymax[i], classes_name[class_num[i]]])


    return result

def draw_results(resized_img, result):

    for i in range(len(result)):
        cv2.rectangle(resized_img, (int(result[i][0]), int(result[i][1])), (int(result[i][2]), int(result[i][3])), (0, 0, 255), thickness=2)
        cv2.putText(resized_img, result[i][4], (int(result[i][0]), int(result[i][1])), 2, 1.5, (0, 0, 255))
    cv2.imshow("predict", resized_img)

common_params = {'image_size': 448, 'num_classes': 20,
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

np_img = cv2.imread('test.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

#saver = tf.train.Saver(net.trainable_collection)
saver = tf.train.Saver()

#saver.restore(sess,'models/pretrain/yolo_tiny.ckpt')
ckpt = tf.train.get_checkpoint_state(os.path.dirname("models/train/checkpoint"))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restore Finished!!")

np_predict = sess.run(predicts, feed_dict={image: np_img})

results = process_predicts(np_predict)
draw_results(resized_img, results)

# cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
# cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
# cv2.imshow("predict", resized_img)
# cv2.imwrite('cat_out.jpg', resized_img)
cv2.waitKey(10000)
sess.close()
