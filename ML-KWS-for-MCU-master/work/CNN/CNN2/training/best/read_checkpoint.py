import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
reader = tf.train.NewCheckpointReader('cnn_2974.ckpt-30')
all_variables = reader.get_variable_to_shape_map()


saver = tf.train.import_meta_graph('cnn_2974.ckpt-30.meta')
with tf.Session() as sess:
    saver.restore(sess,'cnn_2974.ckpt-30')
    writer = tf.summary.FileWriter('log2')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()