"""
Created on Tue Dec 10 12:06:02 2021

@author: HASSINI Houda
"""
import tensorflow as tf
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils, eval_utils
from models import faster_rcnn
from vgg16 import get_model as get_rpn_model
import numpy as np
import dataset

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 1
evaluate = False
use_custom_images = False
custom_image_path = "data/images/"
backbone = args.backbone
io_utils.is_valid_backbone(backbone)
size = 896


hyper_params = train_utils.get_hyper_params(backbone)
classes="/home/admin/Documents/Dev/tf_records/data.names"
path_dataset_test ='/home/admin/Documents/Dev/tf_records/rec_ref_896_int_pha_4_validation_quart.tfrecords'
test_data = dataset.load_tfrecord_dataset(path_dataset_test, classes, size)
labels = ["bg"] + ["s"] + ["p"]
hyper_params["total_labels"] = 3
total_items = 443

img_size = hyper_params["img_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

# if use_custom_images:
#     img_paths = data_utils.get_custom_imgs(custom_image_path)
#     total_items = len(img_paths)
#     test_data = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
#                                                img_paths, img_size, img_size), data_types, data_shapes)
# else:
#     test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, evaluate=evaluate))
# #
test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
for x,y,z in test_data:
    break
#
anchors = bbox_utils.generate_anchors(hyper_params)
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode="inference")
# #
frcnn_model_path = io_utils.get_model_path("faster_rcnn_intensity_only_fold4", backbone)
frcnn_model.load_weights(frcnn_model_path)

step_size = train_utils.get_step_size(total_items, batch_size)
pred_bboxes, pred_labels, pred_scores = frcnn_model.predict(test_data, verbose=1)
"""
if evaluate:
     t = eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
else:
    drawing_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
"""   
stats = {
    "num_good_class_p" : 0,
        "num_bad_class_p" : 0,
            "num_good_class_s"  : 0,
            "num_bad_class_s" : 0,
            "num_missed_p" : 0,
            "num_missed_s" : 0,
            "num_extra_p" : 0,
            "num_extra_s" :0
        }

num_boxes_pred = 0
num_boxes_ref = 0
for batch_id, image_data in enumerate(test_data):
    imgs, gt_boxes, gt_labels = image_data
    start = batch_id * batch_size
    end = start + batch_size
    batch_bboxes, batch_labels, batch_scores = pred_bboxes[start:end], pred_labels[start:end], pred_scores[start:end]
    #denormalized_bboxes = bbox_utils.denormalize_bboxes(batch_bboxes[i], img_size, img_size)
    bb = np.reshape(batch_bboxes,(batch_bboxes.shape[1],4))
    clss = np.reshape(batch_labels,(batch_labels.shape[1]))
    scrs = np.reshape(batch_scores,(batch_scores.shape[1]))

    bb_ref = gt_boxes[0, :, :].numpy()
    clss_ref = gt_labels[0, :, :]
    img = imgs[0, :, :,0].numpy()
    
    num_boxes_pred += bb.shape[0]
    num_boxes_ref+= bb_ref.shape[0]

    found=0
    for index_ref in range(0, bb_ref.shape[0]):
        if bb.shape[0] > 0:
            tensref = np.ones((bb.shape[0], 4)) * bb_ref[index_ref]
            ixmin = np.max((bb[:, 1], tensref[:, 1]), axis=0)
            ixmax = np.min((bb[:, 3], tensref[:, 3]), axis=0)
            iymin = np.max((bb[:, 0], tensref[:, 0]), axis=0)
            iymax = np.min((bb[:, 2], tensref[:, 2]), axis=0)
    
            iw = np.maximum(ixmax - ixmin , 0.)
            ih = np.maximum(iymax - iymin , 0.)
    
            # 2. calculate the area of inters
            inters = iw * ih
    
            # 3. calculate the area of union
            uni = ((bb[:, 2] - bb[:, 0] ) * (bb[:, 3] - bb[:, 1] ) +
                    (tensref[:, 2] - tensref[:, 0] ) * (tensref[:, 3] - tensref[:, 1]  ) -
                    inters)
    
            iou = inters / uni
    
            results = np.max(iou)
            if results >= 0.5:
                index = np.argmax(iou)
                bb = np.delete(bb, (index), axis=0)
                if clss_ref[index_ref]==2 and clss[index]==2:
                    stats["num_good_class_p"]+=1
                    clss = np.delete(clss, (index), axis=0)
                    
                elif clss_ref[index_ref]==2 and clss[index]==1:
                    stats["num_bad_class_p"] += 1
                    clss = np.delete(clss, (index), axis=0)
                    
                elif clss_ref[index_ref] == 1 and clss[index] == 1:
                    stats["num_good_class_s"] += 1
                    clss = np.delete(clss, (index), axis=0)
                    
                elif clss_ref[index_ref] == 1 and clss[index] == 2 :
                    stats["num_bad_class_s"] += 1
                    clss = np.delete(clss, (index), axis=0)
            else:
                if clss_ref[index_ref]==2:
                    stats["num_missed_p"]+=1
                    
                elif clss_ref[index_ref]==1:
                    stats["num_missed_s"] += 1
    if bb.shape[0]>0:
        for index_pred_box in range(0, bb.shape[0]):
            if clss[index_pred_box]==2:
                stats["num_extra_p"] += 1
            elif clss[index_pred_box]==1:
                stats["num_extra_s"] += 1


vp=stats["num_good_class_p"]
fn = stats["num_bad_class_p"]+ stats["num_missed_p"]
fp= stats["num_bad_class_s"]+stats["num_extra_p"]
vn =stats["num_good_class_s"]+ stats["num_extra_s"]+stats["num_missed_s"]
sensitivity=(vp) / (vp + fn)
specificity=(vn) / (fp + vn)
MCC = ((vp * vn) - (fp * fn)) / np.sqrt((vp + fp) * (vp + fn) * (vn + fp ) * (vn + fn))
F1 = vp/(vp + fp/2 + fn/2)

 
