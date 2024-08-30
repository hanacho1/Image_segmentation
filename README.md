## Image segmentation(DeepLabV3Plus-Pytorch)
This project is an example of training and evaluating deep learning models for small object segmentation.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training](#training)
4. [Testing](#testing)

## Overview
This project involves the following key steps:
1. Prepareing custom dataset: Converting label files in JSON format to XML format from images stored in Google Drive.
[Download the dataset from Google Drive]([https://drive.google.com/uc?id=FILE_ID&export=download](https://drive.google.com/drive/folders/1XANY18zT8qBWTSP8uH8tW5PPtoHU-CEH?usp=drive_link](https://drive.google.com/drive/folders/1XANY18zT8qBWTSP8uH8tW5PPtoHU-CEH?usp=drive_link))
2. Training a deep learning model using the converted data.
3. Segmentation small objects in test images using the trained model.

## Create Docker Container
```bash
docker build -t deeplabv3plus .
docker run --shm-size=8g --gpus all -v path/to/desired/directory:/workspace -it --rm deeplabv3plus 
```


## Training
Before starting the training process, data preparation and conversion are necessary.
1. Annotation format conversion
```bash
python update_and_convert_labels.py --input_dir /path/to/json/folder --output_dir /path/to/xml/folder
```
2. Generate mask images
```bash
python generate_mask_image.py
```
3. Creating Custom Data Python Scripts : Reference code `datasets/custom_dataset.py`

4. Training
```bash
python3 main.py --model deeplabv3plus_resnet101 --gpu_id 0,1,2 --dataset custom --lr 0.01 --crop_size 513 --batch_size 4 --output_stride 16  --save_val_results 
```

## Testing
Test images are saved in the 'test_result' folder.
```bash
python3 predict.py --input ./testset --dataset custom --model deeplabv3plus_resnet101 --ckpt checkpoints/best_deeplabv3plus_resnet101_custom_os16.pth --save_val_results_to test_result
```
