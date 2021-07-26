# Bead-Detection-TF2-SSD-MobileNetV2
- This model was constructed based on the pretrained model **SSD MobileNet V2 FPNLite 640x640** in [TF2 Model Zoo!](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- Please use **Data_Preparing_Scropt.py** to generate train.record, validation.record and test.record for model training and testing.
- After finishing model training, please use **bead_detection_no_performance_TF2.py** to predict your data. If you want to know the performance result, please use **bead_detection_model_TF2.py** to predict your data.
- If you want to visualize data with annotation, please use **Visualization_Annotations.py**.
- The samples of visualized images can be seen in **visualized_images directory**.
- The samples of detected images can be seen in **Detection Results**.
