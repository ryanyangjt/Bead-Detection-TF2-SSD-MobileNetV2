import cv2
import xml.etree.ElementTree as ET
import os
import shutil
import glob
import pandas as pd
import itertools
from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
from absl import flags
import tf_record

flags.DEFINE_string('slice_path', './TF_Sliced_Data/', 'Path to output Slice Data')
flags.DEFINE_string('data_dir', './Separate_Data/', 'Path to images')
flags.DEFINE_string('fine_tune_model_path', './fine_tune_model/', 'Path to save fine tune model')
flags.DEFINE_integer('slice_size_x', 480, 'Size of x for sliced data')
flags.DEFINE_integer('slice_size_y', 480, 'Size of y for sliced data')
flags.DEFINE_integer('step_size', 120, 'Size of step_size for sliced data')
FLAGS = flags.FLAGS


def sliced_xml_to_csv(full_img, xml_info, rec_scope, y_num, x_num, d, sliced_directory):
    xmin, ymin, xmax, ymax = rec_scope
    sliced_img = full_img[ymin:ymax, xmin:xmax]
    img_name = xml_info[0][0].split('.jpg')[0]
    cv2.imwrite(sliced_directory + d + '/' + img_name + '_' + str(y_num) + '_' + str(x_num) + '.jpg', sliced_img)

    original_info = (img_name + '_' + str(y_num) + '_' + str(x_num) + '.jpg',
                     xmax - xmin,
                     ymax - ymin,
                     xmin,
                     ymin
                     )

    csv_list = []
    for row in list(xml_info[1:]):
        if xmin <= row[4] <= xmax and xmin <= row[6] <= xmax and ymin <= row[5] <= ymax and ymin <= row[7] <= ymax:
            if row[3] == 'bead':
                class_name = 'bead'
            value = (img_name + '_' + str(y_num) + '_' + str(x_num) + '.jpg',
                     xmax - xmin,
                     ymax - ymin,
                     'bead',
                     row[4] - xmin,
                     row[5] - ymin,
                     row[6] - xmin,
                     row[7] - ymin
                     )
            csv_list.append(value)

    return csv_list, original_info


def xml_to_csv_all(path, file):
    file_name = file.split('.xml')[0]
    xml_list = []
    for xml_file in glob.glob(path + file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member[0].text == 'bead':
                class_name = 'bead'
                bndbox = member.find('bndbox')
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         'bead',
                         int(bndbox[0].text),
                         int(bndbox[1].text),
                         int(bndbox[2].text),
                         int(bndbox[3].text)
                         )
                xml_list.append(value)
    return xml_list


def create_label_maps(classes, start=1):
    content = ''
    for id, name in enumerate(classes, start=start):
        content = content + "item {\n"
        content = content + "    id: " + str(id) + "\n"
        content = content + "    name: '" + name + "'\n}\n\n"
    return content[:-1]


def create_directory(sliced_directory, fine_tune_path, label_maps):
    sliced_train = sliced_directory + "Train/"
    sliced_test = sliced_directory + "Test/"
    sliced_validation = sliced_directory + "Validation/"
    sliced_info = sliced_directory + "Origin_Info/"
    inference_graph = sliced_directory + "inference_graph/"
    if os.path.exists(sliced_directory):
        shutil.rmtree(sliced_directory)
        os.mkdir(sliced_directory)
        os.mkdir(sliced_train)
        os.mkdir(sliced_test)
        os.mkdir(sliced_info)
        os.mkdir(sliced_validation)
        os.mkdir(inference_graph)
    else:
        os.mkdir(sliced_directory)
        os.mkdir(sliced_train)
        os.mkdir(sliced_test)
        os.mkdir(sliced_info)
        os.mkdir(sliced_validation)
        os.mkdir(inference_graph)

    # For Fine Tune Path
    fine_tune_detection_path = fine_tune_path + "detection_results/"
    detection_full_imgs_path = fine_tune_detection_path + "Full_Images/"
    detection_slice_imgs_path = fine_tune_detection_path + "Slice_Images/"
    trained_model_path = fine_tune_path + "trained_model/"

    if os.path.exists(fine_tune_path):
        shutil.rmtree(fine_tune_detection_path)
        shutil.rmtree(trained_model_path)

        os.mkdir(fine_tune_detection_path)
        os.mkdir(detection_full_imgs_path)
        os.mkdir(detection_slice_imgs_path)
        os.mkdir(trained_model_path)
        os.remove(fine_tune_path + 'label_maps.pbtxt')
        with open(fine_tune_path + "label_maps.pbtxt", "w") as f:
            f.write(label_maps)
            f.close()
    else:
        os.mkdir(fine_tune_path)
        os.mkdir(fine_tune_detection_path)
        os.mkdir(detection_full_imgs_path)
        os.mkdir(detection_slice_imgs_path)
        os.mkdir(trained_model_path)
        # Create label_maps.pbtxt
        with open(fine_tune_path + "label_maps.pbtxt", "w") as f:
            f.write(label_maps)
            f.close()


def slice_data(dir_path, slice_size_x, slice_size_y, step_size, sliced_directory):
    for dir in os.listdir(dir_path):
        _dir = dir_path + dir
        img_path = _dir + '/Images/'
        annotation_path = _dir + '/Annotations/'

        data_information = []
        for img_file in os.listdir(img_path):
            print('Slicing Image ' + img_file + '...')
            img = cv2.imread(img_path + img_file)
            f = img_file.split('.jpg')[0]
            xml_list = xml_to_csv_all(annotation_path, f + '.xml')

            count_all = 0
            count_y = 0
            original_information = []
            for y in range(0, img.shape[0] - slice_size_y, slice_size_y - step_size):
                count_x = 0
                for x in range(0, img.shape[1] - slice_size_x, slice_size_x - step_size):
                    slice_result = sliced_xml_to_csv(img, xml_list, [x, y, x + slice_size_x, y + slice_size_y], count_y,
                                                     count_x, dir, sliced_directory)
                    data_information.append(slice_result[0])
                    original_information.append(slice_result[1])
                    count_x += 1
                    count_all += 1
                count_y += 1

            x_initial = 0
            y_initial = 0
            x_total = count_all
            for i in range(count_y):
                slice_result = sliced_xml_to_csv(img, xml_list, [img.shape[1] - slice_size_x, y_initial,
                                                                 img.shape[1], y_initial + slice_size_y], i, count_x,
                                                 dir, sliced_directory)
                data_information.append(slice_result[0])
                original_information.append(slice_result[1])
                y_initial += slice_size_y - step_size
                count_all += 1
            for j in range(int(x_total / count_y)):
                slice_result = sliced_xml_to_csv(img, xml_list,
                                                 [x_initial, img.shape[0] - slice_size_y, x_initial + slice_size_x,
                                                  img.shape[0]], count_y, j, dir, sliced_directory)
                data_information.append(slice_result[0])
                original_information.append(slice_result[1])
                x_initial += slice_size_x - step_size
                count_all += 1

            slice_result = sliced_xml_to_csv(img, xml_list, [img.shape[1] - slice_size_x,
                                                             img.shape[0] - slice_size_y,
                                                             img.shape[1], img.shape[0]],
                                             count_y, count_x, dir, sliced_directory)
            data_information.append(slice_result[0])
            original_information.append(slice_result[1])
            count_all += 1
            print(count_all)

            column_name_origin = ['filename', 'width', 'height', 'xmin', 'ymin']
            original_information_df = pd.DataFrame(original_information, columns=column_name_origin)
            original_information_df.to_csv(sliced_directory + 'Origin_Info/' + f + '_info.csv', index=None)

        data_information = list(itertools.chain.from_iterable(data_information))
        column_name_data = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        data_information_df = pd.DataFrame(data_information, columns=column_name_data)
        data_information_df.to_csv(sliced_directory + dir + '.csv', index=None)


def generate_tfrecord(csv_input, image_dir, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = tf_record.split(examples, 'filename')
    for group in grouped:
        tf_example = tf_record.create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def main(unused_argv):
    # Create Directory
    slice_path = FLAGS.slice_path
    fine_tune_path = FLAGS.fine_tune_model_path
    label_map = create_label_maps(['bead'])
    create_directory(slice_path, fine_tune_path, label_map)

    # Parameter for Slice Data
    slice_size_x = int(FLAGS.slice_size_x)
    slice_size_y = int(FLAGS.slice_size_y)
    step_size = int(FLAGS.step_size)
    data_path = FLAGS.data_dir
    slice_data(data_path, slice_size_x, slice_size_y, step_size, slice_path)

    # Generate TFRecord for Training Model
    train_csv_input = slice_path + 'Train.csv'
    val_csv_input = slice_path + 'Validation.csv'
    test_csv_input = slice_path + 'test.csv'
    train_image_dir = slice_path + 'Train/'
    val_image_dir = slice_path + 'Validation/'
    test_image_dir = slice_path + 'test/'
    train_record_save_path = slice_path + 'train.record'
    validation_record_save_path = slice_path + 'validation.record'
    test_record_save_path = slice_path + 'test.record'

    generate_tfrecord(train_csv_input, train_image_dir, train_record_save_path)
    generate_tfrecord(val_csv_input, val_image_dir, validation_record_save_path)
    generate_tfrecord(test_csv_input, test_image_dir, test_record_save_path)


if __name__ == '__main__':
    tf.compat.v1.app.run()


