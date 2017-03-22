



import tensorflow as tf
import numpy as np
import re

from yolo.net.net import Net 

class YoloTinyNet(Net):

  def __init__(self, common_params, net_params, test=False):
    """
    common params: a params dict
    net_params   : a params dict
    """
    super(YoloTinyNet, self).__init__(common_params, net_params)
    #process params
    self.image_size = int(common_params['image_size'])
    self.num_classes = int(common_params['num_classes'])
    self.cell_size = int(net_params['cell_size'])
    self.boxes_per_cell = int(net_params['boxes_per_cell'])
    self.batch_size = int(common_params['batch_size'])
    self.eval_batch_size = 56
    self.weight_decay = float(net_params['weight_decay'])
    self.boxcode = tf.lin_space(0.05, 0.95, 10)

    if not test:
      self.object_scale = float(net_params['object_scale'])
      self.noobject_scale = float(net_params['noobject_scale'])
      self.class_scale = float(net_params['class_scale'])
      self.coord_scale = float(net_params['coord_scale'])

  def inference(self, images):
    """Build the yolo model

    Args:
      images:  4-D tensor [batch_size, image_height, image_width, channels]
    Returns:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    """
    conv_num = 1

    temp_conv = self.conv2d('conv' + str(conv_num), images, [3, 3, 3, 16], stride=1)
    conv_num += 1

    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 16, 32], stride=1)
    conv_num += 1

    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 32, 64], stride=1)
    conv_num += 1
    
    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
    conv_num += 1

    #temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
    conv_num += 1     

    # temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    # conv_num += 1

    # temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    # conv_num += 1

    predicts = self.conv2d('output', temp_conv, [3, 3, 1024, 801])

    #temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2))

    #Fully connected layer
    # local1 = self.local('local1', temp_conv, self.cell_size * self.cell_size * 1024, 256)
    #
    # local2 = self.local('local2', local1, 256, 4096)
    #
    # local3 = self.local('local3', local2, 4096, self.cell_size * self.cell_size * (self.num_classes + self.boxes_per_cell * 5), leaky=False, pretrain=False, train=True)
    #
    # n1 = self.cell_size * self.cell_size * self.num_classes
    #
    # n2 = n1 + self.cell_size * self.cell_size * self.boxes_per_cell
    #
    # class_probs = tf.reshape(local3[:, 0:n1], (-1, self.cell_size, self.cell_size, self.num_classes))
    # scales = tf.reshape(local3[:, n1:n2], (-1, self.cell_size, self.cell_size, self.boxes_per_cell))
    # boxes = tf.reshape(local3[:, n2:], (-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))
    #
    # local3 = tf.concat(axis=3, values=[class_probs, scales, boxes])
    #
    # predicts = local3

    return predicts

  def iou(self, boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    #calculate the left up point
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    #intersection
    intersection = rd - lu 

    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    return inter_square/(square1 + square2 - inter_square + 1e-6)

  def get_pridict_box(self, coord_code):
    filter_code = tf.round(coord_code)
    x = tf.reduce_max(tf.multiply(tf.slice(filter_code, [0,0,0], [7,7,10]), self.boxcode), axis=2)
    y = tf.reduce_max(tf.multiply(tf.slice(filter_code, [0,0,10], [7,7,10]), self.boxcode), axis=2)
    w = tf.reduce_max(tf.multiply(tf.slice(filter_code, [0,0,20], [7,7,10]), self.boxcode), axis=2)
    h = tf.reduce_max(tf.multiply(tf.slice(filter_code, [0,0,30], [7,7,10]), self.boxcode), axis=2)
    return tf.stack([x, y, w, h],axis=2)


  def cond1(self, num, object_num, loss, predict, label, avg_iou, nilboy):
    """
    if num < object_num
    """
    return num < object_num


  def body1(self, num, object_num, loss, predict, labels, avg_iou, nilboy):
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, w, h, class)
    """
    label = labels[num:num+1, :]
    label = tf.reshape(label, [-1])

    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
    #calculate responsible tensor [CELL_SIZE, CELL_SIZE]
    #max_value = tf.constant(6.99, dtype=tf.float32)
    #center_x = tf.minimum(label[0] / (self.image_size / self.cell_size), max_value)
    center_x = label[0] / (self.image_size / self.cell_size)
    diff_x = center_x - tf.floor(center_x)
    center_x = tf.floor(center_x)

    # center_y = tf.minimum(label[1] / (self.image_size / self.cell_size), max_value)
    center_y = label[1] / (self.image_size / self.cell_size)
    diff_y = center_y - tf.floor(center_y)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size -center_x - 1]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    response = tf.pad(response, temp, "CONSTANT")
    objects = response

    #calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    #predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]
    

    #predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4*10])

    #iou_predict_truth = self.iou(predict_boxes, label[0:4])
    #calculate C [cell_size, cell_size, boxes_per_cell]
    #C = iou_predict_truth * tf.reshape(objects, [self.cell_size, self.cell_size, 1])
    #change to [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    change = tf.ones([self.cell_size, self.cell_size, self.boxes_per_cell])
    C = change * tf.reshape(objects, [self.cell_size, self.cell_size, 1])
    #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = change * tf.reshape(response, (self.cell_size, self.cell_size, 1))
    
    # max_I = tf.reduce_max(I, 2, keep_dims=True)
    #
    # I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

    #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    no_I = tf.ones_like(I, dtype=tf.float32) - I 


    p_C = predict[:, :, -1:]

    P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

    #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:-1]

    #class_loss
    #class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
    #class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

    #object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale
    #object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

    #noobject_loss
    #noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

    #new_coord_loss
    #coord_P = predict[:,:,self.num_classes+self.boxes_per_cell:]

    #change = tf.ones([self.cell_size, self.cell_size, 4])
    #predict_xywh = change * tf.reshape(objects, [self.cell_size, self.cell_size, 1])

    #object_box = self.get_pridict_box(coord_P)

    # predict_xywh = object_box * tf.reshape(objects, [self.cell_size, self.cell_size, 1])
    # predict_xywh = tf.reshape(predict_xywh, [self.cell_size, self.cell_size, 1, 4],name="predict_xywh")
    # tf.Print(predict_xywh, [predict_xywh])
    #
    # iou = self.iou(predict_xywh, label[0:4])
    # iou = tf.reduce_sum(iou)

    thresh_vector = tf.lin_space(0.0, 0.9, 10)

    x_vector = tf.fill([10], diff_x)
    x_label = tf.subtract(x_vector, thresh_vector)
    coord_x_gt = tf.ceil(x_label)

    y_vector = tf.fill([10], diff_y)
    y_label = tf.subtract(y_vector, thresh_vector)
    coord_y_gt = tf.ceil(y_label)

    w_vector = tf.fill([10], label[2] / self.image_size)
    w_label = tf.subtract(w_vector, thresh_vector)
    coord_w_gt = tf.ceil(w_label)

    h_vector = tf.fill([10], label[3] / self.image_size)
    h_label = tf.subtract(h_vector, thresh_vector)
    coord_h_gt = tf.ceil(h_label)

    coord_gt = tf.concat([coord_x_gt, coord_y_gt, coord_w_gt, coord_h_gt], 0)

    class_coord_gt = tf.multiply(coord_gt, tf.slice(P,[0],[1]))
    for k in range(1,20):
        temp = tf.multiply(coord_gt, tf.slice(P,[k],[1]))
        class_coord_gt = tf.concat([class_coord_gt, temp], 0)

    class_coord_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - class_coord_gt))

    #coord_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (coord_P - coord_gt))


    #coord_loss
    # coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
    #              tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size)) +
    #              tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
    #              tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

    nilboy = I

    return num + 1, object_num, [loss[0] + class_coord_loss, loss[1] + object_loss, loss[2] + noobject_loss], predict, labels, avg_iou, nilboy



  def loss(self, predicts, labels, objects_num):
    """Add Loss to all the trainable variables

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    class_loss = tf.constant(0, tf.float32)
    object_loss = tf.constant(0, tf.float32)
    noobject_loss = tf.constant(0, tf.float32)
    coord_loss = tf.constant(0, tf.float32)
    loss = [0, 0, 0]
    temp_iou = tf.constant(0, tf.float32)
    ave_iou = 0
    for i in range(self.batch_size):
      predict = predicts[i, :, :, :]
      label = labels[i, :, :]
      object_num = objects_num[i]
      nilboy = tf.ones([14,14,1])
      tuple_results = tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num, [class_loss, object_loss, noobject_loss], predict, label, temp_iou, nilboy])
      for j in range(3):
        loss[j] = loss[j] + tuple_results[2][j]

      ave_iou = ave_iou + tuple_results[5]
      nilboy = tuple_results[6]

    tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2])/self.batch_size)
    #print (ave_iou)
    tf.summary.scalar('class_coord_loss', loss[0]/self.batch_size)
    tf.summary.scalar('object_loss', loss[1]/self.batch_size)
    tf.summary.scalar('noobject_loss', loss[2]/self.batch_size)
    #tf.summary.scalar('coord_loss', loss[3]/self.batch_size)
    #tf.summary.scalar('avg_iou', ave_iou)
    tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2])/self.batch_size )

    return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy

