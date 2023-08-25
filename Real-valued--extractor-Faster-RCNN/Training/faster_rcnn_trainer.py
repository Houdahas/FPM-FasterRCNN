"""
Created on Tue Dec 14 14:52:06 2021

@author: HASSINI Houda
"""
import os
#s.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import io_utils, data_utils, train_utils, bbox_utils
from models import faster_rcnn
from vgg16 import get_model as get_rpn_model
import dataset



args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 32
epochs = 100

size = 896
img_size = 896
backbone = args.backbone
io_utils.is_valid_backbone(backbone)
hyper_params = train_utils.get_hyper_params("vgg16")
classes = "/home/admin/Documents/Dev/tf_records/data.names"
path_dataset_train = '/home/admin/Documents/Dev/tf_records/rec_ref_896_int_pha_4_train.tfrecords'
train_data = dataset.load_tfrecord_dataset(path_dataset_train, classes, size)
path_dataset_val = '/home/admin/Documents/Dev/tf_records/rec_ref_896_int_pha_4_validation.tfrecords'
val_data = dataset.load_tfrecord_dataset(path_dataset_val, classes, size)


# We add 1 class for background
hyper_params["total_labels"] = 3
train_total_items = 1771
val_total_items = 443

data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
train_data = train_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

anchors = bbox_utils.generate_anchors(hyper_params)
frcnn_train_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
frcnn_val_feed = train_utils.faster_rcnn_generator(val_data, anchors, hyper_params)
# #
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params)
frcnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                    loss=[None] * len(frcnn_model.output))
faster_rcnn.init_model(frcnn_model, hyper_params)



# # If you have pretrained rpn model
# You can load rpn weights for faster training
# rpn_load_weights = True
# if rpn_load_weights:
#     rpn_model_path = io_utils.get_model_path("rpn", backbone)
#     rpn_model.load_weights(rpn_model_path)
# Load weights
frcnn_model_path = io_utils.get_model_path("faster_rcnn_intensity_phase_fold4", backbone)
log_path = io_utils.get_log_path("faster_rcnn_intensity_phase_fold4", backbone)
#load_weights = True
#if load_weights:
#frcnn_model.load_weights(frcnn_model_path) #log_path = io_utils.get_log_path("faster_rcnn", backbone)

checkpoint_callback = ModelCheckpoint(frcnn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=log_path)
step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)
frcnn_model.fit(frcnn_train_feed, verbose =1,
                steps_per_epoch=step_size_train,
                validation_data =frcnn_val_feed,
                validation_steps=step_size_val,
                epochs=epochs,
                #class_weight = class_weights,
                callbacks=[checkpoint_callback, tensorboard_callback])
