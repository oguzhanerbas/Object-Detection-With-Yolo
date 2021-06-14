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
#### System Properties<br>
%cat /etc/lsb/-release<br>
#### Storage Update <br>
!apt-get update<br>
#### Learning adress<br>
%pwd (output must be ‘/content’)<br>
#### Unzip<br>
!unzip “/content/drive/My Drive/custom_object_model/darknet.zip”<br>
#### Setting Adress<br>
%cd /content/darknet<br>
#### For Checking<br>
%pwd (output must be the ‘content/darknet’)<br>
#### Download dus2unix<br>
!sodu apt install dos2unix<br>
#### Transform the files to Unix<br>
!find . -type f -print0 | xargs -0 dos2unix<br>
#### Permission<br>
!chmod +x /content/darknet<br>
#### Delete File<br>
!rm /content/darknet/backup -r<br>
#### Linking with Drive and Backup<br>
!ln -s /content/drive/”My Drive”/object_weights/backup /content/darknet<br>
#### Starting Train<br>
!./darknet detector train object_data/object.data object_yolov4.cfg yolov4.conv.137 -map -dont_show<br>

## Python Code

    # -*- coding: utf-8 -*-
    """
    Created on Wed May 26 11:46:18 2021

    @author: Oguzhan
    """

    import cv2
    import numpy as np

    img = cv2.imread("images/1430189615.jpg")

   ### The size of the picture has been saved
    img_width = img.shape[1]
    img_height = img.shape[0]

  ### Setting of yolov4 size
    img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)

  ### Classes what we use in the video
    labels = ["smooth_leaf"]

  ### Colors what we use for rectangle
    colors = ["0,255,255"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

  ### Yolo's datas for find mask
    model = cv2.dnn.readNetFromDarknet("YOLO Files/leaf_yolov4.cfg", "YOLO Files/leaf_yolov4_last.weights")

  ### This part of code take matrix from yolos data and identity the object and add to lists
    layers = model.getLayerNames()

    output_layer = [layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(img_blob)

    detection_layers = model.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidences_list = []

  ### This part measured the object size
    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

  ### Object cover up with rectangle
            if confidence > 0.3:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    for max_id in max_ids:
        max_class_id = max_id[0]
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label = "{}: {:.2f}%".format(label, confidence * 100)
        print("predicted object {}".format(label))

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 1)
        cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

   ### Image show with rectangle
    cv2.imshow("Detection Window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
![alt text](https://github.com/oguzhanerbas/Object-Detection-With-Yolo/blob/main/leaf_detected_image.png)
 
