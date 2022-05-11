# from keras_segmentation.models.unet import vgg_unet

# model = vgg_unet(n_classes=6 ,  input_height=256, input_width=256  )

from keras_segmentation.models.unet import mobilenet_unet

model = mobilenet_unet(n_classes=6 ,  input_height=256, input_width=256  )

# from keras_segmentation.models.unet import resnet50_unet

# model = resnet50_unet(n_classes=6 ,  input_height=256, input_width=256  )

# from keras_segmentation.models.segnet import mobilenet_segnet

# model = mobilenet_segnet(n_classes=6 ,  input_height=256, input_width=256  )

# from keras_segmentation.models.segnet import resnet50_segnet

# model = resnet50_segnet(n_classes=6 ,  input_height=256, input_width=256  )

model.train(
    train_images =  "/home/suraj/Desktop/Major_Project/Custom Dataset/Combined/Train Images/images1",
    train_annotations = "/home/suraj/Desktop/Major_Project/Custom Dataset/Combined/Train Images/annotated1",
    checkpoints_path = "/home/suraj/Desktop/Major_Project/tmp/vgg_unet_1" , 
    batch_size=8,
    auto_resume_checkpoint=True,
    epochs=5
)

model.save("/home/suraj/Desktop/Major_Project/saved_model/mobilenet_unet.h5")
