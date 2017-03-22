import sys
import os
import csv
import math

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
code_weight = np.arange(0.05, 1.0, 0.1)

f = open("./2007_val.txt", "r")
data = csv.reader(f)

def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    a = box1[2] * box1[3]
    b = box2[2] * box2[3]
    resut = float(intersection) / float(a+b-intersection)
    return resut

def get_wh(w, h):
    result_w = np.max(np.multiply(w, code_weight), axis=1)
    result_h = np.max(np.multiply(h, code_weight), axis=1)
    return result_w, result_h

def get_xy(x, y):
    result_x = np.max(np.multiply(x, code_weight), axis=1)
    result_y = np.max(np.multiply(y, code_weight), axis=1)
    return result_x, result_y

def process_predicts_new(predicts, width, height):
    class_type = 20
    coord_layer_num = 40
    side_num = 14

    classes_coord = predicts[0, :, :, 0:class_type*coord_layer_num]
    C = predicts[0, :, :, class_type*coord_layer_num:]
    threshold = 0.1
    classes_coord = np.reshape(classes_coord, (side_num, side_num, class_type, coord_layer_num))
    C = np.reshape(C, (side_num, side_num, 1))

    p_classes_center_x = classes_coord[:, :, :, :10]
    p_classes_center_y = classes_coord[:, :, :, 10:20]
    p_classes_w = classes_coord[:, :, :, 20:22]
    p_classes_h = classes_coord[:, :, :, 30:32]
    p_classes_center_x = np.max(p_classes_center_x, axis=3)
    p_classes_center_y = np.max(p_classes_center_y, axis=3)
    p_classes_w = np.mean(p_classes_w, axis=3)
    p_classes_h = np.mean(p_classes_h, axis=3)
    p_classes = p_classes_center_x * p_classes_center_y * p_classes_w * p_classes_h

    P = C * p_classes

    filter_mat_probs = np.array(P > threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    class_num = filter_mat_boxes[2]
    probs_filtered = P[filter_mat_probs]

    #coordinate = np.reshape(classes_coord[:,:,filter_mat_boxes[2],:], (7, 7, 40))

    filted_coordinate = np.array(classes_coord > 0.5, dtype='int')

    max_coordinate = filted_coordinate[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2],:]

    xcenter = max_coordinate[:,:10]
    ycenter = max_coordinate[:,10:20]
    w = max_coordinate[:,20:30]
    h = max_coordinate[:,30:40]

    w, h = get_wh(w, h)
    xcenter, ycenter = get_xy(xcenter, ycenter)
    xcenter = (filter_mat_boxes[1] + xcenter) * (width/14.0)
    ycenter = (filter_mat_boxes[0] + ycenter) * (height/14.0)

    w = w * width
    h = h * height

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0

    xmax = xmin + w
    ymax = ymin + h

    result = []
    for i in range(len(class_num)):
        for j in range(i+1, len(class_num)):
            if iou([xmin[i], ymin[i], xmax[i], ymax[i]],[xmin[j], ymin[j], xmax[j], ymax[j]]) > 0.5 and\
                    (class_num[i] == class_num[j]) and class_num[i] != -1:
                xmin[i],ymin[i],xmax[i],ymax[i] = min(xmin[i], xmin[j]),min(ymin[i],ymin[j]),max(xmax[i], xmax[j]), max(ymax[i], ymax[j])
                class_num[j]=-1

    for i in range(len(class_num)):
        if class_num[i] != -1:
            result.append([xmin[i], ymin[i], xmax[i], ymax[i], classes_name[class_num[i]], probs_filtered[i]])


    return result




def process_predicts(predicts, width, height):
    class_type = 20
    boxes_per_cell = 1
    p_classes = predicts[0, :, :, 0:class_type]
    C = predicts[0, :, :, class_type:class_type+boxes_per_cell]
    coordinate = predicts[0, :, :, class_type+boxes_per_cell:]
    threshold = 0.2

    p_classes = np.reshape(p_classes, (7, 7, 1, class_type))
    C = np.reshape(C, (7, 7, boxes_per_cell, 1))

    P = C * p_classes

    filter_mat_probs = np.array(P > threshold, dtype='bool')

    filter_mat_boxes = np.nonzero(filter_mat_probs)

    probs_filtered = P[filter_mat_probs]

    #print P[5,1, 0, :]

    # index = np.argmax(P)
    #
    # index = np.unravel_index(index, P.shape)

    class_num = filter_mat_boxes[3]

    coordinate = np.reshape(coordinate, (7, 7, boxes_per_cell, 40))

    filted_coordinate = np.array(coordinate > 0.5, dtype='int')

    max_coordinate = filted_coordinate[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2], :]

    xcenter = max_coordinate[:,:10]
    ycenter = max_coordinate[:,10:20]
    w = max_coordinate[:,20:30]
    h = max_coordinate[:,30:40]

    w, h = get_wh(w, h)
    xcenter, ycenter = get_xy(xcenter, ycenter)

    xcenter = (filter_mat_boxes[1] + xcenter) * (width/7.0)
    ycenter = (filter_mat_boxes[0] + ycenter) * (height/7.0)

    w = w * width
    h = h * height

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0

    xmax = xmin + w
    ymax = ymin + h

    result = []
    for i in range(len(class_num)):
        for j in range(i+1, len(class_num)):
            if iou([xmin[i], ymin[i], xmax[i], ymax[i]],[xmin[j], ymin[j], xmax[j], ymax[j]]) > 0.5 and\
                    (class_num[i] == class_num[j]) and class_num[i] != -1:
                xmin[i],ymin[i],xmax[i],ymax[i] = min(xmin[i], xmin[j]),min(ymin[i],ymin[j]),max(xmax[i], xmax[j]), max(ymax[i], ymax[j])
                class_num[j]=-1

    for i in range(len(class_num)):
        if class_num[i] != -1:
            result.append([xmin[i], ymin[i], xmax[i], ymax[i], classes_name[class_num[i]], probs_filtered[i]])


    return result

def draw_results(resized_img, result):

    for i in range(len(result)):
        cv2.rectangle(resized_img, (int(result[i][0]), int(result[i][1])), (int(result[i][2]), int(result[i][3])), (0, 0, 255), thickness=2)
        cv2.putText(resized_img, result[i][4], (int(result[i][0]), int(result[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imshow("predict", resized_img)
    cv2.imwrite('output.jpg', resized_img)

def write_results(results, img_name):

    for i in range(len(results)):
        filename = "./results/comp4_det_test_" + str(results[i][4]) + ".txt"
        with open(filename, 'a') as resultfile:
            resultfile.write(img_name + " " + '%.6f' % results[i][5] + " " + '%.6f' % results[i][0] + " " + '%.6f' % results[i][1] \
                             + " " + '%.6f' % results[i][2] + " " + '%.6f' % results[i][3] + "\n")

def make_file():
    for i in range(len(classes_name)):
        filename = "./results/comp4_det_test_" + str(classes_name[i]) + ".txt"
        with open(filename, "w") as resultfile:
            resultfile.write("")

def plot_images(predicts):

    _ ,_ ,_ , layer_num = predicts.shape
    #rows = math.ceil(layer_num / 10.0)
    rows = 4

    # for k in range(20):
    #     fig = plt.figure()
    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    #     for i in range(40):
    #         a = fig.add_subplot(rows, 10, i+1)
    #         imgplot = plt.imshow(predicts[0,:,:,k*40+i], norm=norm)
    #         # if i < len(classes_name):
    #         #     a.set_title(classes_name[i])
    #     fig.subplots_adjust(right=0.8)
    #
    #     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig.colorbar(imgplot,ticks=[0.1, 0.3, 0.5, 0.7, 0.9] ,cax=cbar_ax)
    #     plt.title(classes_name[k])
    #     plt.show()

    fig = plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    imgplot = plt.imshow(predicts[0, :, :, -1], norm=norm)
    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(imgplot, ticks=[0.1, 0.3, 0.5, 0.7, 0.9], cax=cbar_ax)
    #plt.show()


common_params = {'image_size': 448, 'num_classes': 20,
                'batch_size':1}
net_params = {'cell_size': 14, 'boxes_per_cell':1, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

#saver = tf.train.Saver(net.trainable_collection)
saver = tf.train.Saver()

#saver.restore(sess,'models/pretrain/yolo_tiny.ckpt')
ckpt = tf.train.get_checkpoint_state(os.path.dirname("models/train/checkpoint"))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restore Finished!!")

make_file()

for row in data:
    img = cv2.imread(row[0])
    height, width, _ = img.shape
    resized_img = cv2.resize(img, (448, 448))
    img_name = row[0].split("/")[-1].split(".")[0]
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


    np_img = np_img.astype(np.float32)

    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))

    np_predict = sess.run(predicts, feed_dict={image: np_img})
    cv2.imwrite('temp.jpg', img)
    plot_images(np_predict)

    results = process_predicts_new(np_predict, width, height)

    # results = process_predicts(np_predict, width, height)
    draw_results(img, results)
    # write_results(results, img_name)

    cv2.waitKey(1000)

# cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
# cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
# cv2.imshow("predict", resized_img)
# cv2.imwrite('cat_out.jpg', resized_img)
cv2.waitKey(10000)
sess.close()
