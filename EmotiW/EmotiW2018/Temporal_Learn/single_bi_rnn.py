import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from util import Util as util
from keras.utils import np_utils

class DataSet(object):

  def __init__(self,
               images,
               labels):
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

ROOT_PATH = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/Temporal_Learn/'
TRAIN_PATH = '/home/dmsl/nas/DMSL/AFEW/NpzData/Train_only/'
VAL_PATH = '/home/dmsl/nas/DMSL/AFEW/NpzData/Val/'

# TODO : select rnn cell (GRU / LSTM)
#CELL = 'bi_lstm'
CELL = 'bi_gru'

# TODO: Direcotry name
name = 'train_only_512'

# TODO : check gpu
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "1"
config.gpu_options.allow_growth = True

# TODO : check save path
DIR_NAME = CELL + '_' + name
SAVE_PATH = ROOT_PATH + DIR_NAME
MODEL_PATH = SAVE_PATH + '/' + DIR_NAME + '.ckpt'

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# TODO : check hyperparameter
training_steps = 25000
batch_size = 128
display_step = 200

num_input = 1664
timesteps = 16
num_hidden = 512
num_classes = 7
dropout = 0.5

print("Load data...")
X_TRAIN_PATH = TRAIN_PATH + 'x_DenseNet169_original_frames_overlap.npz'
Y_TRAIN_PATH = TRAIN_PATH + 'y_DenseNet169_original_frames_overlap.npz'
X_VAL_PATH = VAL_PATH + 'x_DenseNet169_original_frames_no_overlap.npz'
Y_VAL_PATH = VAL_PATH + 'y_DenseNet169_original_frames_no_overlap.npz'

x_train, x_val = util.load_from_npz(X_TRAIN_PATH), util.load_from_npz(X_VAL_PATH)
y_train, y_val = util.load_from_npz(Y_TRAIN_PATH), util.load_from_npz(Y_VAL_PATH)

y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)
x_frame_data = DataSet(x_train, y_train)

print("Build Model")
X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
lr_placeholder = tf.placeholder(tf.float32)

def Bi_RNN(x, weight, bias):
    x = tf.unstack(x, timesteps, axis=1)

    if CELL == 'bi_lstm':
        cell_fw = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        cell_bw = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        dropout_fw_cell = rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        dropout_bw_cell = rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
    elif CELL == 'bi_gru':
        cell_fw = rnn.GRUCell(num_hidden)
        cell_bw = rnn.GRUCell(num_hidden)
        dropout_fw_cell = rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        dropout_bw_cell = rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

    outputs, _, _ = tf.nn.static_bidirectional_rnn(dropout_fw_cell, dropout_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

weight = tf.Variable(tf.truncated_normal([2*num_hidden, num_classes], stddev=0.1), name='W')
bias = tf.Variable(tf.truncated_normal([num_classes]), name='b')

logits = Bi_RNN(X, weight, bias)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_placeholder)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

pred_sum = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

# save checkpoint
saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
print("Start learning!")
with tf.Session(config=config) as sess:

    # Run the initializer
    sess.run(init)
    max_acc = 0
    for step in range(1, training_steps+1):
        
        if step < 5000:
            learning_rate = 1e-3
        elif step > 5000 and step < 10000:
            learning_rate = 1e-4
        elif step > 10000 and step < 15000:
            learning_rate = 1e-5
        elif step > 20000:
            learning_rate = 1e-6
            
        batch_x, batch_y = x_frame_data.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y, keep_prob: dropout, lr_placeholder: learning_rate}
        sess.run(train_op, feed_dict=feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

        if step % (x_train.shape[0] // batch_size) == 0:
            test_data = x_val.reshape((-1, timesteps, num_input))
            test_label = y_val
            num_true = sess.run(pred_sum, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0, lr_placeholder: learning_rate})
            test_acc = num_true / x_val.shape[0]

            if max_acc < test_acc:
                print("Save " + DIR_NAME + " - performance improved from {:.6f} to {:.6f}".format(max_acc, test_acc))
                max_acc = test_acc
                saver.save(sess, MODEL_PATH)

    print("Optimization Finished!")
    print("Final Test Accuracy : {}".format(max_acc))

# save configuration
save_dict = {
    'Test_Accuracy' : str(max_acc),
    'Cell_type' : CELL,
    'Learning_rate' : str(learning_rate),
    'Batch_size' : str(batch_size),
    'Hidden_unit' : str(num_hidden),
    'Dropout' : str(dropout),
    'X_train' : X_TRAIN_PATH.split('/')[-1],
    'Y_train' : Y_TRAIN_PATH.split('/')[-1],
    'X_test' : X_VAL_PATH.split('/')[-1],
    'Y_test' : Y_VAL_PATH.split('/')[-1]
}
with open(SAVE_PATH + '/config.txt', 'w') as f:
    for key, content in sorted(save_dict.items()):
        f.write(key + ' : ' + content + '\n')