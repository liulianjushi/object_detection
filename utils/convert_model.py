import tensorflow as tf
from tensorflow.python.saved_model import signature_constants


def write_saved_model(frozen_graph_model, saved_model_path):
    frozen_graph_def = tf.Graph()
    with frozen_graph_def.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
            tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(image_tensor)}
            tensor_info_outputs = {}
            for k, v in tensor_dict.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs=tensor_info_inputs,
                outputs=tensor_info_outputs,
                method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: detection_signature, }, )
            builder.save()


if __name__ == '__main__':
    PATH_TO_FROZEN_GRAPH = ""
    saved_model = ""
    write_saved_model(PATH_TO_FROZEN_GRAPH, saved_model)
