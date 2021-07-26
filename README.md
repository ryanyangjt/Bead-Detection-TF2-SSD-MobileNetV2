# Directory Structure
```bash
├── fine_tune_model
│   ├── ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8
│   ├── trained_model
│   ├── detection_results
│   │   ├── Full_Images
│   │   └── Slice_Images
│   └── label_maps.pbtxt
├── Separate_Data
│   ├── Train
│   │   ├── Annotations
│   │   └── Images
│   ├── Validation
│   │   ├── Annotations
│   │   └── Images
│   └── Test
│   │   ├── Annotations
│   │   └── Images
├── TF_Sliced_Data
│   ├── inference_graph
│   ├── Origin_Info
│   ├── Train
│   ├── Validation
│   ├── Test
│   ├── train.record
│   ├── validation.record
│   └── test.record
├── visualized_images
├── Data_Preparing_Script.py
├── bead_detection_model_TF2.py
├── bead_detection_no_perfomance_TF2.py
├── model_main_tf2.py
├── exporter_main_v2.py
├── tf_record.py
└── Visualization_Annotations.py
```

# Bead-Detection-TF2-SSD-MobileNetV2
- This model was constructed based on the pretrained model **SSD MobileNet V2 FPNLite 640x640** in [TF2 Model Zoo!](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- Please put your data/annotations in **Separate_Data** directory for train, validation and test.
- Please use **Data_Preparing_Script.py** to generate train.record, validation.record and test.record for model training and testing. 
  ```bash
  python Data_Preparing_Script.py --slice_path=./TF_Sliced_Data/ --data_dir=./Separate_Data/ --fine_tune_model_path=./fine_tune_model/ --slice_size_x=640 --slice_size_y=640 --step_size=100
  ```
- After finishing data preparing procedure, please use **model_main_tf2.py** to train your model.
  ```bash
  python model_main_tf2.py --eval_on_train_data=True --sample_1_of_n_eval_examples=2 --pipeline_config_path=./fine_tune_model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config --model_dir=./fine_tune_model/trained_model/ --alsologtostderr
  ```
- After finishing model training, please use the following command to export your model.
  ```bash
  python exporter_main_v2.py --trained_checkpoint_dir=./fine_tune_model/trained_model/ --output_directory=./TF_Sliced_Data/inference_graph/ --pipeline_config_path=./fine_tune_model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config
  ```
- Please use **bead_detection_no_performance_TF2.py** to predict your data. If you want to know the performance result, please use **bead_detection_model_TF2.py** to predict your data.
- If you want to visualize data with annotation, please use **Visualization_Annotations.py**.
- The samples of visualized images can be seen in **visualized_images directory**.
- The samples of detected images can be seen in **Detection Results**.


