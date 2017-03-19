



import tensorflow as tf
import numpy as np
import re
import sys
import time
import os
from datetime import datetime
from tensorflow.python import debug as tf_debug

from yolo.solver.solver import Solver

hooks = [tf_debug.LocalCLIDebugHook()]

class YoloSolver(Solver):
  """Yolo Solver
  """
  def __init__(self, dataset, net, common_params, solver_params):
    #process params
    self.moment = float(solver_params['moment'])
    self.learning_rate = float(solver_params['learning_rate'])
    self.batch_size = int(common_params['batch_size'])
    self.height = int(common_params['image_size'])
    self.width = int(common_params['image_size'])
    self.max_objects = int(common_params['max_objects_per_image'])
    self.pretrain_path = str(solver_params['pretrain_model_path'])
    self.train_dir = str(solver_params['train_dir'])
    self.max_iterators = int(solver_params['max_iterators'])
    #
    self.dataset = dataset
    self.net = net
    #construct graph
    self.construct_graph()

  def _train(self):
    """Train model

    Create an optimizer and apply to all trainable variables.

    Args:
      total_loss: Total loss from net.loss()
      global_step: Integer Variable counting the number of training steps
      processed
    Returns:
      train_op: op for training
    """
    learning_rate = self.learning_rate
    # starter_learning_rate = self.learning_rate
    # end_learning_rate = self.learning_rate / 1000.0
    # decay_steps = 10000
    # learning_rate = tf.train.polynomial_decay(starter_learning_rate, self.global_step,
    #                                           decay_steps, end_learning_rate,
    #                                           power=0.5)

    tf.summary.scalar('learning rate', learning_rate)
    #opt = tf.train.MomentumOptimizer(learning_rate, self.moment)
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(self.total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

    return apply_gradient_op

  def construct_graph(self):
    # construct graph
    self.global_step = tf.Variable(0,dtype=tf.int32 ,trainable=False, name='global_step')
    self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
    self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
    self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

    self.predicts = self.net.inference(self.images)
    self.total_loss, self.nilboy = self.net.loss(self.predicts, self.labels, self.objects_num)


    tf.summary.scalar('loss', self.total_loss)
    self.train_op = self._train()

  def solve(self):
    #saver1 = tf.train.Saver(self.net.pretrained_collection)
    #saver1 = tf.train.Saver(self.net.trainable_collection)
    saver2 = tf.train.Saver()
    init =  tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session()
    sess.run(init)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    #saver1.restore(sess, self.pretrain_path)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("models/train/checkpoint"))
    if ckpt and ckpt.model_checkpoint_path:
       saver2.restore(sess, ckpt.model_checkpoint_path)
       print ("Restore Finished!!")
    else:
       sess.run(init)


    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
    start_step = self.global_step.eval(sess)
    print ("start_step")
    print (start_step)

    for step in range(start_step, self.max_iterators):
      start_time = time.time()
      np_images, np_labels, np_objects_num = self.dataset.batch()

      _, loss_value, nilboy = sess.run([self.train_op, self.total_loss, self.nilboy], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
      #loss_value, nilboy = sess.run([self.total_loss, self.nilboy], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})


      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = self.dataset.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

        sys.stdout.flush()
      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
        summary_writer.add_summary(summary_str, step)
      if step % 500 == 0:
        saver2.save(sess, self.train_dir + '/model.ckpt', global_step=step)
    sess.close()
