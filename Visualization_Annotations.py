import cv2
import xml.etree.ElementTree as ET
import os
import shutil


visual_images_directory = "./visualized_images/"
if os.path.exists(visual_images_directory):
    shutil.rmtree(visual_images_directory)
    os.mkdir(visual_images_directory)
else:
    os.mkdir(visual_images_directory)

Annotations_Path = "./Annotations/"
xml_files = os.listdir(Annotations_Path)

Images_Path = "./Blur_Images/"
image_files = os.listdir(Images_Path)
print(image_files)

total_good_bead = 0
total_bad_bead = 0
total_dust = 0
for file_id in range(len(image_files)):

    xml = ET.parse(Annotations_Path + xml_files[file_id])
    image = os.path.join(Images_Path, image_files[file_id])
    img = cv2.imread(image)

    root = xml.getroot()
    objs = root.findall('object')

    print("Processing image " + Images_Path + '/' + image_files[file_id] + "...")

    image_name = image_files[file_id].split('.jpg')
    for obj_id in range(len(objs)):
        name = objs[obj_id].find('name').text
        bndbox = objs[obj_id].find('bndbox')
        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)
        if name == 'g_bead':
            total_good_bead += 1
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(img, name, (xmin, ymin-10), 1, 1, (0, 255, 0))
        elif name == 'dust':
            total_dust += 1
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.putText(img, name, (xmin, ymin-10), 1, 1, (255, 0, 0))
        elif name == 'b_bead':
            total_bad_bead += 1
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(img, name, (xmin, ymin-10), 1, 1, (0, 0, 255))
    cv2.imwrite(visual_images_directory + image_name[0] + '.jpg', img)

print("--------------------------------------------------------------")
print("Finish!")
print("Total number of good bead: ", total_good_bead)
print("Total number of bad bead: ", total_bad_bead)
print("Total number of dust: ", total_dust)
