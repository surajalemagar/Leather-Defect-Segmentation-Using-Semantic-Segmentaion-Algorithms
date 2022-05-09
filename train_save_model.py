from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=6 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "/home/suraj/Desktop/Major_Project/Custom Dataset/Combined/Train Images/images1",
    train_annotations = "/home/suraj/Desktop/Major_Project/Custom Dataset/Combined/Train Images/annotated1",
    checkpoints_path = "/home/suraj/Desktop/Major_Project/tmp/vgg_unet_1" , epochs=5
)

model.save("/home/suraj/Desktop/Major_Project/saved_model/vgg_unet.h5")
