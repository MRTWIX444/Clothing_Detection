import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=6)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
np.argmax(predictions[0])

probability_model.save_weights('model_final.h5')


# make Xa prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    plt.imshow(img)
    plt.show()
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


#model1 = load_model('model_final.h5')

# load an image and predict the class
def run_example():
    # load the image
    img = load_image(filename)
    # load model
    # predict the class
    result = probability_model.predict(img)
    print(class_names[np.argmax(result[0])])
    result=class_names[np.argmax(result[0])]
    Label(new, text=result, font=('Helvetica 17 bold')).pack(pady=50)

    #print(result[0])


#front end
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import glob
import cv2
from keras.preprocessing import image

root= tk.Tk()

#Make a Canvas (i.e, a screen for your project

canvas = tk.Canvas(root, width = 750, height = 650)
canvas.configure(bg='grey19')
canvas.pack()

# App title label

label1 = tk.Label(root, bg="blue", fg="white", text=' Clothes Detection System ', font=("Helvetica", 25), borderwidth=3, relief="solid")
canvas.create_window(375, 75, window=label1)


#Open file explorer

def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("JPG", "*.jpg*"),("all files", "*.*")))
    open_win(filename)
#Open a new window for user

import matplotlib.pyplot as plt


def open_win(image):
    global new
    new= Toplevel(canvas)
    new.geometry("1000x850")
    new.title("New Window")
    Label(new, text="Selected image: ", font=('Helvetica 17 bold')).pack(pady=20)

    #create run button
    button2 = tk.Button (new, text='Run', font=("ROG FONTS", 10), bg='green', command = run_example)
    button2.pack(pady=15)

   #inserting user image:
    frame = Frame(new, width=400, height=400)
    frame.pack()
    frame.place(anchor='center', relx=0.5, rely=0.5)
    img = ImageTk.PhotoImage(Image.open(image))
    label = tk.Label(new, image = img).pack()
    new.mainloop()
   

#upload image

button1 = tk.Button (root, text='Upload Image', font=("ROG FONTS", 10), bg='green', command = browseFiles)
canvas.create_window(375, 150, window=button1)



#insert fireimage

frame = Frame(canvas, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)
img = ImageTk.PhotoImage(Image.open("C:/Users/HP/Desktop/dataset/Clothes.jpg"))
label = Label(frame, image = img)
label.pack()

root.mainloop()
