# Leather-Defect-Detection-Using-Convolutional-Neural-Network

## Requirements
1. tensorflow==2.4.1
2. keras==2.4.3
3. opencv-python
4. Pillow
5. tk==0.1.0


## Working on Google Collab
https://colab.research.google.com/drive/147R5_b6-ye-nAyyCNZxEypnKVx0ItnNf

## Train and Save model in h5 format

from keras_segmentation.models.unet import mobilenet_unet

model = mobilenet_unet(n_classes=6 ,  input_height=256, input_width=256  )

model.train(
    train_images =  "/home/suraj/Desktop/Major_Project/Custom Dataset/Combined/Train Images/images1",
    train_annotations = "/home/suraj/Desktop/Major_Project/Custom Dataset/Combined/Train Images/annotated1",
    checkpoints_path = "/home/suraj/Desktop/Major_Project/tmp/vgg_unet_1" , 
    batch_size=8,
    auto_resume_checkpoint=True,
    epochs=5
)

model.save("/home/suraj/Desktop/Major_Project/saved_model/mobilenet_unet.h5")

