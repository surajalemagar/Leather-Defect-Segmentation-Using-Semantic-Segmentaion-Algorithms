from keras_segmentation.models.unet import mobilenet_unet

model = mobilenet_unet(n_classes=6 ,  input_height=256, input_width=256  )

model.load_weights("/home/suraj/Desktop/Major_Project/saved_model/mobilenet_unet.h5")

print(model.evaluate_segmentation( inp_images_dir="/home/suraj/Desktop/Major_Project/Public Dataset/Test Image/images"  ,
                                   annotations_dir="/home/suraj/Desktop/Major_Project/Public Dataset/Test Image/annotated image" ) )
