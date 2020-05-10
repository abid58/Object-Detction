# Object-Detction ECE498SMA SP2020 Final Project
Improved Object Detection in Crowded Traffic Scenes. We tried to implement paper 1, although it became more like statndard YOLO in paper 2. 


## Paper 1:
Visibility Guided NMS: Efficient Boosting of Amodal Object Detection in Crowded Traffic Scenes by Nils Gählert, Niklas Hanselmann, Uwe Franke, Joachim Denzler (Mercedes-Benz AG, R&D)

## Paper 2
You Only Look Once: Unified, Real-time Object Detection by Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. 

### To run this code:
1. download YOLOv3 weights by 'wget https://pjreddie.com/media/files/yolov3.weights' in 'darknet' folder. 
2. For image: python image_yolo.py --input images/example.jpeg --output output_images/example.jpg  --path darknet
3. For video: python video_yolo.py --input videos/example.mp4 --output output_videos/example.avi  --path darknet 

#### Please cite this repo if you use code from here.

#### My teammates:
1. Alexander Krysl
2. jasvir Virdi

