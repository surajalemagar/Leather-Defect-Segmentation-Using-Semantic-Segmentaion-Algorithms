from keras_segmentation.models.unet import mobilenet_unet

model = mobilenet_unet(n_classes=6 ,  input_height=256, input_width=256  )

model.load_weights("/home/suraj/Desktop/Major_Project/saved_model/mobilenet_unet.h5")

out = model.predict_segmentation(
    inp="/home/suraj/Desktop/Major_Project/Public Dataset/Test Image/images/1.png",
    out_fname="/home/suraj/Desktop/Major_Project/output/output1.png"
)
# see output figure in output folder