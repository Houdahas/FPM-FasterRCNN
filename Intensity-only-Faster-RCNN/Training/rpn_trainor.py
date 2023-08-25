import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import io_utils, data_utils, train_utils, bbox_utils
from vgg16 import get_model
import dataset


args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 2
epochs = 400
image_size=896
size = image_size

backbone = args.backbone
io_utils.is_valid_backbone(backbone)

hyper_params = train_utils.get_hyper_params("vgg16")
classes="/home/admin/Documents/Dev/fasterrcnn2/data/tf_records/data.names"
path_dataset_train='/home/admin/Documents/Dev/data/tf_records/rec_ref_896_int_pha_train.tfrecords'
train_data = dataset.load_tfrecord_dataset(path_dataset_train, classes, size)
# path_dataset_val='/home/hassinihouda/Documents/fasterrcnn/data/tf_records/rec_ref_896_int_pha_validation.tfrecords'
# val_data = dataset.load_tfrecord_dataset(path_dataset_val, classes, size)
for x, y, z in train_data:
    break 
hyper_params["total_labels"] = 3
train_total_items = 2
val_total_items = 192

data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
train_data = train_data.padded_batch(batch_size,padded_shapes=data_shapes, padding_values=padding_values)
# val_data = val_data.padded_batch(batch_size,padded_shapes=data_shapes, padding_values=padding_values)


anchors = bbox_utils.generate_anchors(hyper_params)
rpn_train_feed = train_utils.rpn_generator(train_data, anchors, hyper_params)
rpn_val_feed = train_utils.rpn_generator(train_data, anchors, hyper_params)


rpn_model, _ = get_model(hyper_params)
rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                  loss=[train_utils.reg_loss, train_utils.rpn_cls_loss])
rpn_model_path = io_utils.get_model_path("rpn", backbone)

checkpoint_callback = ModelCheckpoint(rpn_model_path, monitor="val_loss", save_best_only=False, save_weights_only=True)
step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)

rpn_model.fit(rpn_train_feed, verbose =1,
              steps_per_epoch = step_size_train,
              # validation_data = rpn_val_feed,
              # validation_steps = step_size_val,
              epochs = 100,
              callbacks = [checkpoint_callback])