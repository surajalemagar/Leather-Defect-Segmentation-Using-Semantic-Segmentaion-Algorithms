from cgi import test
from distutils.command.upload import upload
import imp
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

my_w = tk.Tk()
my_w.geometry("1200x450")
my_w.title('Leather Defect Detection')
my_font1= ('times',18, 'bold')
b1=tk.Button(my_w,text="Select Original Image and Mask Image ",width=50,command=lambda:upload_file())
b1.grid(row=0,column=0,columnspan=4)
def upload_file():
    l=[]
    from keras_segmentation.models.unet import mobilenet_unet
    model2 = mobilenet_unet(n_classes=6 ,  input_height=256, input_width=256  )
    model2.load_weights("/home/suraj/Desktop/Major_Project/saved_model/mobilenet_unet.h5")
    f_types = [('PNG files',"*.png"),('Jpg files','*.jpg'),('All Files',"*.*")]

    # Select Orginal Defect Images
    filename1 = askopenfilename(filetypes=f_types)  
    print("Original Image Selected")
    out = model2.predict_segmentation(
        inp=filename1,
        out_fname="/home/suraj/Desktop/Major_Project/output/output1.png"
        )
    l.append(filename1)
    
    # Select corresponding mask images of defect image
    filename2 = askopenfilename(filetypes=f_types)
    l.append(filename2)
    print("Mask Image selected")

    l.append("/home/suraj/Desktop/Major_Project/output/output1.png")
    col =1
    print("Displaying Segmented Image")
    for f in l:
        img = Image.open(f)
        img = img.resize((380,380))
        img = ImageTk.PhotoImage(img)
        e1=tk.Label(my_w)
        e1.grid(row=3,column=col)
        e1.image=img
        e1['image']=img
        col=col+1
my_w.mainloop()