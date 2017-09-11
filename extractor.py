import os
import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile


BOTTLENECK_TENSOR_NAME = 'mixed_7/join:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

JPG_DIR = ''
ROOT_DIR = 'C:\\Users\\vzs\\OneDrive\\Documents\\small_classifier'
MODEL_DIR = os.path.join(ROOT_DIR, 'model')

def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(
                MODEL_DIR, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                            BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                            RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def compute_btneck(sess, image_path, bottleneck_tensor, image_data_tensor):
    image_data = gfile.FastGFile(image_path, 'rb').read()
    print('image loaded...')
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    print('btneck computed')
    bottleneck_values = np.squeeze(bottleneck_values)
    print('squeezed')
    return bottleneck_values


def generate_bottlenecks(file_list, dst_dir):
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())
    sess = tf.Session(graph=graph)
    print('Loaded model...')
    for img_path in file_list:
        bottleneck_values = compute_btneck(sess,
                                           img_path,
                                           bottleneck_tensor,
                                           jpeg_data_tensor)
        print(bottleneck_values.shape)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        dst_path = os.path.join(dst_dir, img_path.split('.')[-1]+'.txt')
        with open(dst_path, 'w') as dst_file:
            dst_file.write(bottleneck_string + '\n')

if __name__ == '__main__':
    root_dir = ROOT_DIR
    img_path = os.path.join(root_dir, 'trainspotting.jpg')
    generate_bottlenecks([img_path], root_dir)
