# Object-Detection-With-Yolo
Object Detection With YoloV4 and Using MakeSense, Darknet, Google Colaboratory

### Install
- Firstly download from https://git-scm.com/
- Open a new file and enter the file after that download darknet from this link https://github.com/AlexeyAB/darknet also download “yolov4.conv.137” and “yolov4.weights” files
- Find yolov4.cfg file in the darknet file open this file with txt

### Change this value:  
subdivisions = 64, weight = 416, height = 416, max_batches = 2000, steps = 1800(max_batches%90),  classes = 1(for a object detection if you detect one more change the class according to number of object), filters = 18 (classes + 5 * 3) chance (filters location on the activation=linear) change classes and filter two more times.

### Open Images File and Find Image What You Train
- Save in order the file name 1.jpg, 2.jpg … 400.jpg it is so important for training, also file extension must be .jpg. You should find 300-400 images for good training.

### MakeSense
- We use MakeSense for detect the object from images. Go to this link https://www.makesense.ai/
- And click Get Started. Drop the images file and click the detection detect the object from images and Export Annotations(YOLO Format) forward the txt file to images file.

### Training and Testing File
- Create a training file (txt) and write %80 path of images and create test file and write the other path of images (you can change the training and testing rate.

### Creating .named and .data File
- Create txt file and write 
classes = 1
train = leaf_data\leaf_training.txt
valid = leaf_data\leaf_testing.txt
names = leaf_data\leaf.names
backup = backup
-save as “object_name.data”

- Create another txt file and write 
smooth_leaf
save as “object_name.names”

- All this file send to a file named object_data like this <br>
![alt text](https://github.com/oguzhanerbas/Object-Detection-With-Yolo/blob/main/image1.png) <br>
![alt text](https://github.com/oguzhanerbas/Object-Detection-With-Yolo/blob/main/image2.png)

- Send object_data to darknet and transform darknet file to zip

### Google Drive 
- Upload the darknet.zip to Google Drive
- Open new file in Drive named custom_object_model and send zip file to custom_object_model, and create another open new file in named object_weights, again open new file in object_weights named backup.

### Google Colab
Open a new Colab file in drive <br>
Write These Code:<br>
System Properties ==>  %cat /etc/lsb/-release<br>
Storage Update ==> !apt-get update<br>
Learning adress ==>%pwd (output must be ‘/content’)<br>
Unzip ==> !unzip “/content/drive/My Drive/custom_object_model/darknet.zip”<br>
Setting Adress ==> %cd /content/darknet<br>
For Checking ==> %pwd (output must be the ‘content/darknet’)<br>
Download dus2unix ==> !sodu apt install dos2unix<br>
Transform the files to Unix ==> !find . -type f -print0 | xargs -0 dos2unix<br>
Permission ==> !chmod +x /content/darknet<br>
Delete File ==> !rm /content/darknet/backup -r<br>
Linking with Drive and Backup ==> !ln -s /content/drive/”My Drive”/object_weights/backup /content/darknet<br>
Starting Train ==> !./darknet detector train object_data/object.data object_yolov4.cfg yolov4.conv.137 -map -dont_show<br>

