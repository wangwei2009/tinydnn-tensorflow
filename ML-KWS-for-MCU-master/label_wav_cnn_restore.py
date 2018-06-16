# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import nn_inference

import numpy as np

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  #reader = tf.train.NewCheckpointReader('work/CNN/CNN2/training/best/cnn_2974.ckpt-30')
  reader = tf.train.NewCheckpointReader('work/CNN/CNN2_bn_no_gammabeta/training/best/cnn_2987.ckpt-30')
  all_variables = reader.get_variable_to_shape_map()

  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    Reshape_1_op = tf.get_default_graph().get_operation_by_name('Reshape_1').outputs[0]
    Conv2D_op = tf.get_default_graph().get_operation_by_name('Conv2D').outputs[0]
    bn1_op = tf.get_default_graph().get_operation_by_name('bn1/FusedBatchNorm').outputs[0]
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    Reshape_1_output, = sess.run(Reshape_1_op, {input_layer_name: wav_data})
    Conv2D_output, = sess.run(Conv2D_op, {input_layer_name: wav_data})
    mfcc = Reshape_1_output.reshape(1, 49, 10, 1)

    conv_w = reader.get_tensor('Variable')
    conv_bias = reader.get_tensor('Variable_1')
    bn1_moving_mean = reader.get_tensor('bn1/moving_mean')
    bn1_moving_variance = reader.get_tensor('bn1/moving_variance')
    # bn1_gamma = reader.get_tensor('bn1/gamma')
    # bn1_beta = reader.get_tensor('bn1/beta')
    variance_epsilon = 0.0000001

    Conv2D_diy_out = nn_inference.Conv2D(mfcc,conv_w,conv_bias,padding='VALID')

    #bn1_diy_out = bn1_beta+ bn1_gamma*(Conv2D_diy_out-bn1_moving_mean)/np.sqrt(bn1_moving_variance+variance_epsilon)
    bn1_diy_out = nn_inference.bn(x=Conv2D_diy_out,
                                 moving_mean=reader.get_tensor('bn1/moving_mean'),
                                 moving_variance=reader.get_tensor('bn1/moving_variance')
                                 )
    bn1_output, = sess.run(bn1_op, {input_layer_name: wav_data})

    relu_diy_out = nn_inference.ReLu(bn1_diy_out)

    Conv2D_1_diy_out = nn_inference.Conv2D(relu_diy_out,reader.get_tensor('Variable_2'),reader.get_tensor('Variable_3'),strides=[1,2,1,1],padding='VALID')
    Conv2D_1_output, = sess.run(tf.get_default_graph().get_operation_by_name('Conv2D_1').outputs[0], {input_layer_name: wav_data})

    bn2_diy_output = nn_inference.bn(x=Conv2D_1_diy_out,
                                 moving_mean=reader.get_tensor('bn2/moving_mean'),
                                 moving_variance=reader.get_tensor('bn2/moving_variance')
                                 )
    bn2_output, = sess.run(tf.get_default_graph().get_operation_by_name('bn2/FusedBatchNorm').outputs[0], {input_layer_name: wav_data})

    Relu_1_diy_output = nn_inference.ReLu(bn2_diy_output)
    Relu_1__output , = sess.run(tf.get_default_graph().get_operation_by_name('Relu_1').outputs[0], {input_layer_name: wav_data})

    Reshape_2_output = Relu_1__output.reshape(1,-1)
    Reshape_2_diy_output = Relu_1_diy_output.reshape(1,-1)

    MatMul_output  = sess.run(tf.get_default_graph().get_operation_by_name('MatMul').outputs[0], {input_layer_name: wav_data})
    MatMul_diy_output = nn_inference.fc(x=Reshape_2_diy_output,
                                        W=reader.get_tensor('W'),
                                        bias=reader.get_tensor('b')
                                        )

    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
