# coding: utf-8
import os
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import warnings
import pandas as pd
import xml.etree.ElementTree as ET


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_SAVED_MODEL = "./TF_Sliced_Data/inference_graph/saved_model"

print('Loading model...', end='')

# Calculate loading model time
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


PATH_TO_LABELS = "./fine_tune_model/label_maps.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

# Path information
PATH_TO_ALL_TEST_IMG_DIR = './Separate_Data/Test/Images/'
PATH_TO_ALL_TEST_SLICED_IMG_DIR = './TF_Sliced_Data/Test/'
PATH_TO_ALL_INFO_CSV_DIR = './TF_Sliced_Data/Origin_Info/'
PATH_TO_SAVE_DETECTION_SLICED_IMG_RESULT_DIR = \
    './fine_tune_model/detection_results/Slice_Images/'
PATH_TO_SAVE_DETECTION_FULL_IMG_RESULT_DIR = \
    './fine_tune_model/detection_results/Full_Images/'

testing_imgs = os.listdir(PATH_TO_ALL_TEST_IMG_DIR)
testing_sliced_imgs = os.listdir(PATH_TO_ALL_TEST_SLICED_IMG_DIR)
print(testing_imgs)

im = cv2.imread(PATH_TO_ALL_TEST_SLICED_IMG_DIR + testing_sliced_imgs[0])
image_height, image_width, _ = im.shape

for test_img in testing_imgs:
    test_img_id = test_img.split('.jpg')[0]
    print(test_img_id)
    predicted_boxes = []
    predicted_scores = []

    start_time = time.time()
    for sliced_img in testing_sliced_imgs:
        sliced_img_id = sliced_img.split('_')[0]
        if sliced_img_id == test_img_id:
            image_path = os.path.join(PATH_TO_ALL_TEST_SLICED_IMG_DIR, sliced_img)
            img_name = sliced_img.split('.jpg')[0]

            image_np = load_image_into_numpy_array(image_path)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)

            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                           for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            data = pd.read_csv(PATH_TO_ALL_INFO_CSV_DIR + test_img_id + '_info.csv', usecols=['filename', 'xmin', 'ymin'])
            data = data.to_dict()
            dic_info = {}
            for f in range(len(data['filename'])):
                dic_info[data['filename'][f]] = [data['xmin'][f], data['ymin'][f]]

            x_origin, y_origin = dic_info[img_name+'.jpg']

            draw_img = cv2.imread(image_path)
            box = detections['detection_boxes']
            score = detections['detection_scores']
            for idx in range(len(box)):
                delta_y = int((box[idx][2] - box[idx][0]) * image_height)
                delta_x = int((box[idx][3] - box[idx][1]) * image_height)
                area = delta_y * delta_x
                max_area_threshold = 110*110
                min_are_threshold = 70*70
                score_threshold = 0.4
                if score[idx] >= score_threshold and min_are_threshold <= area <= max_area_threshold:
                    y_min_abs = int(box[idx][0] * image_height) + y_origin
                    x_min_abs = int(box[idx][1] * image_width) + x_origin
                    y_max_abs = int(box[idx][2] * image_height) + y_origin
                    x_max_abs = int(box[idx][3] * image_width) + x_origin
                    predicted_boxes.append([y_min_abs, x_min_abs, y_max_abs, x_max_abs])
                    predicted_scores.append(score[idx])

            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                  image_np_with_detections,
                  detections['detection_boxes'],
                  detections['detection_classes'],
                  detections['detection_scores'],
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=200,
                  min_score_thresh=.40,
                  agnostic_mode=False)

            cv2.imwrite(PATH_TO_SAVE_DETECTION_SLICED_IMG_RESULT_DIR + img_name +
                        '_detected.jpg', image_np_with_detections)

    axis_window_checked = np.asarray(predicted_boxes, dtype=float)
    score_info_checked = np.asarray(predicted_scores, dtype=float)

    # non_maximum_suppression
    selected_indices = tf.image.non_max_suppression(
        boxes=axis_window_checked, scores=score_info_checked, iou_threshold=0.5, score_threshold=0.4, max_output_size=200)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds for detecting'.format(elapsed_time))

    # draw
    selected_boxes = tf.gather(axis_window_checked, selected_indices)
    selected_scores = tf.gather(score_info_checked, selected_indices)
    full_img = cv2.imread(PATH_TO_ALL_TEST_IMG_DIR + test_img)
    Annotations_Path = 'D:/TF_Cetification_Test_Env/Separate_Data/Test/Annotations/'
    xml = ET.parse(os.path.join(Annotations_Path, test_img_id + '.xml'))
    root = xml.getroot()
    objs = root.findall('object')
    total_bead = 0
    gt_boxes = []
    gt_scores = []
    for obj_id in range(len(objs)):
        name = objs[obj_id].find('name').text
        if name == 'bead':
            bndbox = objs[obj_id].find('bndbox')
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
            gt_boxes.append([ymin, xmin, ymax, xmax])
            gt_scores.append(1)
            total_bead += 1
            cv2.rectangle(full_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(full_img, 'bead', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
    cv2.putText(full_img, '1. # Ground Truth Beads (Red Boxes): ' + str(total_bead), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 255), 2)
    cv2.putText(full_img, '2. # Predicted Beads (Green Boxes): ' + str(len(selected_boxes)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 255), 2)
    for bbox, score_bbox in zip(selected_boxes, selected_scores):
        y, x, y2, x2 = bbox
        sc = score_bbox
        cv2.rectangle(full_img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(full_img, 'bead: ' + str(int(sc*100)) + '%', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Performance
    def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    detected_boxes_gt = []
    iou_for_boxed = []
    for gt_b in gt_boxes:
        max_iou = 0
        box_iou = []
        for detected_b in selected_boxes:
            iou = float(bb_intersection_over_union(gt_b, detected_b))
            if iou > max_iou:
                max_iou = iou
                box_iou = detected_b
        if max_iou >= 0.7:
            detected_boxes_gt.append(box_iou)
            iou_for_boxed.append(round(max_iou, 2))
    print(len(detected_boxes_gt))
    for bbb, iou_value in zip(detected_boxes_gt, iou_for_boxed):
        y, x, y2, x2 = bbb
        # cv2.rectangle(full_img, (int(x), int(y)), (int(x2), int(y2)), (255, 206, 135), 2)
        cv2.putText(full_img, 'IoU: ' + str(iou_value), (int(x), int(y) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 206, 135), 2)

    cv2.putText(full_img, '3. # Detected Beads: ' + str(len(detected_boxes_gt)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 255), 2)
    cv2.putText(full_img, 'Accuracy (3/2): ' + str(round(len(detected_boxes_gt)/len(selected_boxes), 2)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 255), 2)
    cv2.putText(full_img, 'Accuracy (3/1): ' + str(round(len(detected_boxes_gt)/len(gt_boxes), 2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 255), 2)
    cv2.imwrite(PATH_TO_SAVE_DETECTION_FULL_IMG_RESULT_DIR + test_img_id + '_detected.jpg', full_img)

