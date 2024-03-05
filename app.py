#! usr/bin/python3

import base64
import shutil
import cv2
import usb.core
import usb.util
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image, ImageTk
from itertools import product
import tkinter as tk
import customtkinter as ctk
from CTkListbox import *
from CTkMessagebox import *
import math
from datetime import datetime
import subprocess
import re


# Address enum
READ_CHIP_ID                    = 54 # 0x36
START_GET_IMAGE_TRANSFER        = 83 # 0x53

GET_OPERATION_MODE              = 61 # 0x3D
GET_IMAGE_PROCESSING_MODE       = 63 # 0x3F
GET_FIRMWARE_INFO               = 78 # 0x4E
GET_FACTORY_SETTINGS            = 88 # 0x58


SET_OPERATION_MODE              = 60 # 0x3C
SET_IMAGE_PROCESSING_MODE       = 62 # 0x3E
SET_FIRMWARE_INFO_FEATURES      = 85 # 0x55
SET_FACTORY_SETTINGS_FEATURES   = 86 # 0x56

WIDTH = 320
HEIGHT = 240
RAW_WIDTH = 342
RAW_HEIGHT = 260

URL = "http://localhost:5000"
CAP_IMG_PATH = "/home/inc472thermal/program/app/images/savedimage.jpg"
CACHE_IMG_PATH = "/home/inc472thermal/program/app/cache/"
TILE_IMG_PATH="/home/inc472thermal/program/app/tile/"
DATABASE_PATH="/home/inc472thermal/program/app/database/"
STITCH_PATH="/home/inc472thermal/program/app/stitched"

stream_image = None
captured = None

camera_state = False

def sorted_alphanumeric(data):
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  return sorted(data, key=alphanum_key)

class App():

  def __init__(self, window: ctk.CTk):
    self.window = window
    self.window.title("App")

    self.container = ctk.CTkFrame(self.window)

    self.container.pack(side="top", fill="both", expand=True)

    self.container.grid_rowconfigure(0, weight=1)
    self.container.grid_columnconfigure(0, weight=1)

    self.frames = {}

    self.menu = tk.Menu(self.window)

    self.file_menu = tk.Menu(self.menu, tearoff=0)
    self.file_menu.add_command(label="Capture",command=lambda:self.show_frame(CapturePage))
    self.file_menu.add_command(label="Files",command=lambda:self.show_frame(FilesPage))

    self.menu.add_cascade(label="Camera", menu=self.file_menu)

    self.settings = tk.Menu(self.menu, tearoff=0)

    self.settings.add_command(label="Adjustments", command=lambda:self.show_frame(AdjustmentPage), state='disabled')

    self.menu.add_cascade(label="Settings", menu=self.settings)

    self.window.config(menu=self.menu)

    for F in (CapturePage, FilesPage, AdjustmentPage):
        frame = F(self.container, self.window)
        self.frames[F] = frame
        frame.grid(row=0, column=0, sticky="nsew")

    self.show_frame(CapturePage)

  def show_frame(self, cont):

    frame = self.frames[cont]
    frame.tkraise()

#   *****   PAGES   *****
    
class CapturePage(ctk.CTkFrame):

  def __init__(self, parent, controller):
    self.window = controller
    ctk.CTkFrame.__init__(self, parent)
    label = ctk.CTkLabel(self, text="Capture Page", font=("default", 20))
    label.pack(pady=10, padx=10)
    self.status_var = tk.StringVar()
    self.status_var.set("Camera Off!")
    self.status = ctk.CTkLabel(self, textvariable=self.status_var)
    self.status.pack(padx=10, pady=10, anchor=tk.CENTER)
    self.current_image = None
    self.seek = None
    self.stream = None
    self.capture_page_canvas = ctk.CTkCanvas(self, width=WIDTH, height=HEIGHT)
    self.capture_page_canvas.pack()
    self.space = ctk.CTkLabel(self, text="").pack()
    self.capture_page_b1 = ctk.CTkButton(self, text= 'Start', command=self.start_stream)
    self.capture_page_b1.pack(padx=10, pady=5)
    self.capture_page_b2= ctk.CTkButton(self, text= 'Stop', command=self.stop_stream)
    self.capture_page_b2.pack(padx=10, pady=5)
    self.capture_page_b3 = ctk.CTkButton(self, text= 'Capture', command=lambda: self.button_pressed(self.captureAndProcess))
    self.capture_page_b3.pack(padx=10, pady=5)

    self.update_cam()

  def start_stream(self):
    global camera_state
    if camera_state is False:  
      try:
        self.seek = SeekPro()
        self.stream = Stream(self.seek)
        camera_state = True  
        self.status_var.set("Camera On!")
      except usb.core.USBTimeoutError:
        camera_state = False
        self.start_stream()
    
  def stop_stream(self):
    global camera_state
    global stream_image
    if camera_state is True:
      self.stream.stop()
      self.current_image = None
      self.seek = None
      self.stream = None
      camera_state = False
      stream_image = None
      self.status_var.set("Camera Off!")

  def button_pressed(self, command):
    self.status_var.set("Capturing and processing images...")
    command()
    self.status_var.set("Camera On!")

  def update_cam(self):
    global camera_state
    
    if camera_state is True:
      self.stream.start() 

    if stream_image is not None:
      self.current_image = Image.fromarray(stream_image)
      self.photo = ImageTk.PhotoImage(image=self.current_image)
      self.capture_page_canvas.create_image(0,0, image=self.photo, anchor=tk.NW)
    else:
      self.current_image = Image.fromarray(np.zeros((HEIGHT, WIDTH))) 
      self.photo = ImageTk.PhotoImage(image=self.current_image)
      self.capture_page_canvas.create_image(0,0, image=self.photo, anchor=tk.NW)

    self.window.after(15,self.update_cam)

  def captureAndProcess(self):
    global camera_state
    if captured is not None and camera_state is True:
      try:
        for listName in os.listdir(TILE_IMG_PATH):
          if os.path.isfile(os.path.join(TILE_IMG_PATH, listName)):
            os.remove(os.path.join(TILE_IMG_PATH,listName))

        for listName in os.listdir(CACHE_IMG_PATH):
          if os.path.isfile(os.path.join(CACHE_IMG_PATH, listName)):
            os.remove(os.path.join(CACHE_IMG_PATH,listName))

        print("Capturing image...")
        with open(CAP_IMG_PATH,"wb") as fh:
          fh.write(base64.decodebytes(captured)) 
        print("Captured Successfully!")
        try:
          self.detectAndCrop()
        except NameError:
          print("Error: something went wrong with detecting and croping the image the image{0}".format(NameError))
        try:
          self.sliceAndTile()
        except:
          print("Error: There is a problem in silcing and tiling the images")
        try:
          self.img_rescale()
        except:
          print("There is a problem in the rescaling stage")
        try:
          for listName in sorted_alphanumeric(os.listdir(TILE_IMG_PATH)):
            ProcessImage(os.path.join(TILE_IMG_PATH, listName), listName)
        except:
          print("There is a problem with Processing the images")
        try:
          self.stitch_images()
        except:
          print("There is a problem stitching the images")
        try:
          self.save_image()
        except:
          print("There is a problem in the message box")
      except NameError:
        print("Error: Captured Unsucessfully! {0}".format(NameError))
    else:
      print("Error:There is no image to capture..")  

  def sliceAndTile(self): 
    filename = "savedCropedImg.jpg"
    dir_in = "/home/inc472thermal/program/app/images/"
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    image_list = np.array([w,h])
    d_min = np.argmin(image_list)
    d = int(math.floor(image_list[d_min]/3))

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    count = 0
    for i, j in grid:
      box = (j, i, j+d, i+d)
      out = os.path.join(TILE_IMG_PATH, f'{name}{count}{ext}')
      img.crop(box).save(out)
      count += 1

  def detectAndCrop(self):

    print("Starting to detect object and crop relevant image") 

    input_image = cv2.imread(CAP_IMG_PATH, cv2.IMREAD_UNCHANGED)

    if len(input_image.shape) == 2:
      gray_input_image = input_image.copy()
    else:
      gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    upper_threshold, thresh_input_image = cv2.threshold(
      gray_input_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lower_threshold = 0.5 * upper_threshold

    canny = cv2.Canny(input_image, lower_threshold, upper_threshold)

    pts = np.argwhere(canny > 0)

    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    output_image = input_image[y1:y2, x1:x2]

    output_path = "/home/inc472thermal/program/app/images/savedCropedImg.jpg"
    
    cv2.imwrite(output_path, output_image)
    print("Successfully saved savedCropedImg.jpg")

    if os.path.isfile(CAP_IMG_PATH):
      print("removing CAP_IMG_PATH")
      os.remove(CAP_IMG_PATH) 
      print("Successfully removed path")   

  def stitch_images(self):
    images = []
    split_images = []

    for listName in sorted_alphanumeric(os.listdir(CACHE_IMG_PATH)):
      path = os.path.join(CACHE_IMG_PATH, listName)
      print(path)
      img = Image.open(path)
      images.append(img) 

    length = int(len(images)/3)

    width = images[0].width * (length)
    height = images[0].height * 3

    img_width = images[0].width
    img_height = images[0].height

    for i in range(0, 3):
      split_images.append(images[length*i:length*(i+1)])
      
    stitched_image = Image.new('RGB', (width, height))

    for i in range(0,3):
      for j in range(0, length):
        stitched_image.paste(split_images[i][j], (j*img_width, i*img_height))
    
    stitched_image.save(f'{STITCH_PATH}/stitchedImage.jpg')

    for i in images:
      i.close()

    for i in range(0,3):
      for j in range(0, length):
        split_images[i][j].close()

  def save_image(self):
    dialog = CTkMessagebox(title="Save Image", message="Do you want to save the processed images?", icon="question", option_1="Yes", option_2="No")

    response = dialog.get()

    if response == "Yes":
      try:
        now = datetime.now()
        now = f"{now}"
        parent = DATABASE_PATH
        path = os.path.join(parent, now)
        os.mkdir(path)

        print("Created directory successfully!")

        shutil.copy(os.path.join(STITCH_PATH, "stitchedImage.jpg"), path)

      except:
        print("Unsucessful creating directory!")  

    
  def img_rescale(self): 

    for listName in os.listdir(TILE_IMG_PATH):
      resized = cv2.resize(cv2.imread(os.path.join(TILE_IMG_PATH, listName)), (128, 128), interpolation=cv2.INTER_AREA)
      cv2.imwrite(os.path.join(TILE_IMG_PATH, listName), resized)
    
    print("Finished Rescaling All Images")

class FilesPage(ctk.CTkFrame):

  def __init__(self, parent, controller: ctk.CTk):
    self.window = controller
    self.parent = parent
    ctk.CTkFrame.__init__(self, parent)
    label = ctk.CTkLabel(self, text="Files Page", font=("default", 20))
    label.pack(pady=10, padx=10)
    self.ref_button = ctk.CTkButton(self, text="Refresh", command=self.update_list)
    self.ref_button.pack(padx=10, pady=10)
    self.listbox = CTkListbox(self, command=self.selected_value)
    self.listbox.pack(fill="both", expand=True, padx=10, pady=10)
    self.getting_list()

  def update_list(self):
    self.window.after(0, self.getting_list) 

  def specimen_do(self, path, value, index):
    dialog = CTkMessagebox(title=f"Specimen {value}",message="What do you want to do with this specimen", icon="question", option_1="Open file", option_2="Delete file")

    response = dialog.get()

    if response == "Open file":
      subprocess.Popen(['xdg-open', path])
    elif response == "Delete file":
      if os.path.exists(path):
        self.listbox.delete(index=index)
        shutil.rmtree(path)
        self.update_list()
    
  def getting_list(self):
    i: int = 0
    self.listbox.delete("all")
    for listName in os.listdir(DATABASE_PATH):
      if os.path.isdir(os.path.join(DATABASE_PATH, listName)):
        name: str = f"{listName}"
        self.listbox.insert(i, name)
        i+=1
        
  def selected_value(self, value):
    path: str = f"{DATABASE_PATH}{value}"
    index = self.listbox.curselection()
    if os.path.exists(path):
      self.specimen_do(path, value, index)
    else:
      self.listbox.delete(index=index)
      print("This path does not exist")
     

class AdjustmentPage(ctk.CTkFrame):

  def __init__(self, parent, controller):
    ctk.CTkFrame.__init__(self, parent)
    label = ctk.CTkLabel(self, text="Adjustment Page", font=("default", 20))
    label.pack(pady=10, padx=10)

    

#   *****   Detecting Pitting Corrosion   *****

class ProcessImage:
  def __init__(self, path, name):
    self.interpreter = tf.lite.Interpreter("/home/inc472thermal/program/app/model/model2.tflite")
    self.max_num_classes = 1
    self.image_path = path
    self.name = name
    self.boxes = None
    self.scores = None
    self.classes = None   
    self.image_np = None

    self.model_interpretation()
    self.processed_image()

  def model_interpretation(self):

    #reshape the input to match the size of image
    self.interpreter.resize_tensor_input(0, [1, 128, 128, 3], strict=True)

    #allocate and set tensor
    self.interpreter.allocate_tensors()
    _, height, width, _ = self.interpreter.get_input_details()[0]['shape']

    input_details = self.interpreter.get_input_details()
    self.image_np = np.array(cv2.imread(self.image_path))
    input_tensor = (self.image_np).astype(np.uint8)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    #set input tensor as np.unit8 of image.

    self.interpreter.set_tensor(input_details[0]['index'], input_tensor)
    self.interpreter.invoke()

    # Get the output tensor
    output_details = self.interpreter.get_output_details()
    output_tensor = self.interpreter.get_tensor(output_details[0]['index'])#raw_detection_boxes
    input_shape = self.interpreter.get_input_details()[0]['shape']
    self.boxes = self.interpreter.get_tensor(output_details[4]['index'])[0]
    self.classes = self.interpreter.get_tensor(output_details[5]['index'])[0]
    self.scores = self.interpreter.get_tensor(output_details[6]['index'])[0]
    self.classes = self.classes.astype(int)

  # Assuming you have these variables from your original code
  # boxes, classes, scores, category_index

  def processed_image(self):

    self.image_np = np.array(cv2.imread(self.image_path))
    self.image_np = cv2.cvtColor(self.image_np.copy(), cv2.COLOR_BGR2RGB)
    minScore = 0.4

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
    ax.imshow(self.image_np)

    # Iterate over each detection and draw bounding box if score is above minScore
    for box, score, class_id in zip(self.boxes, self.scores, self.classes):
      if score >= minScore:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * self.image_np.shape[1])
        xmax = int(xmax * self.image_np.shape[1])
        ymin = int(ymin * self.image_np.shape[0])
        ymax = int(ymax * self.image_np.shape[0])
        box_coords = (xmin, ymin), xmax - xmin, ymax - ymin
        color = 'y'  # You can customize the color here
        text_color = 'k'
        rect = patches.Rectangle(*box_coords, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        ax.text(xmin, ymin - 5, f'{score:.2f}', color=text_color, backgroundcolor = color, fontsize=8, ha='left', va='bottom')

    # Hide the axes
    ax.axis('off')

    # Show the image
    plt.savefig(f"/home/inc472thermal/program/app/cache/{self.name}", bbox_inches="tight", pad_inches=0.0)
    print(f"Succesfully Saved {self.name}")


#   *****   Seek Camera   *****

class SeekPro():

  def __init__(self):
    self.dev = usb.core.find(idVendor=0x289d, idProduct=0x0011)
    if not self.dev:
      raise IOError('Device not found')
    
    usb.util.dispose_resources(self.dev)
    self.dev.reset()

    self.calib = None
    for i in range(5):
      if i == 4:
        print("Could not get the dead pixels frame!")
        self.dead_pixels = []
        break
      self.init()
      status,ret = self.grab()
      if status == 4:
        self.dead_pixels = self.get_dead_pix_list(ret)
        break

  def stop(self):
    usb.util.dispose_resources(self.dev)

  def get_dead_pix_list(self,data):

    img = self.crop(np.frombuffer(data,dtype=np.uint16).reshape(
          RAW_HEIGHT,RAW_WIDTH))
    return list(zip(*np.where(img<100)))

  def correct_dead_pix(self,img):

    for i,j in self.dead_pixels:
      img[i,j] = np.median(img[max(0,i-1):i+2,max(0,j-1):j+2])
    return img

  def crop(self,raw_img):

    return raw_img[4:4+HEIGHT,1:1+WIDTH]

  def send_msg(self,bRequest,  data_or_wLength,
      wValue=0, wIndex=0,bmRequestType=0x41,timeout=None):
  
    assert (self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex,
      data_or_wLength, timeout) == len(data_or_wLength))

  def receive_msg(self,bRequest, data, wValue=0, wIndex=0,bmRequestType=0xC1,
      timeout=None):

    return self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex,
          data, timeout)

  def deinit(self):
 
    for i in range(3):
        self.send_msg(0x3C, b'\x00\x00')

  def init(self):

    self.send_msg(SET_OPERATION_MODE, b'\x00\x00')
 
    self.send_msg(SET_FACTORY_SETTINGS_FEATURES, b'\x06\x00\x08\x00\x00\x00')
 
    self.send_msg(SET_FIRMWARE_INFO_FEATURES,b'\x17\x00')

    self.send_msg(SET_FACTORY_SETTINGS_FEATURES, b"\x01\x00\x00\x06\x00\x00")

    for i in range(10):
      for j in range(0,256,32):
        self.send_msg(
            SET_FACTORY_SETTINGS_FEATURES,b"\x20\x00"+bytes([j,i])+b"\x00\x00")

    self.send_msg(SET_FIRMWARE_INFO_FEATURES,b"\x15\x00")
  
    self.send_msg(SET_IMAGE_PROCESSING_MODE,b"\x08\x00")

    self.send_msg(SET_OPERATION_MODE,b"\x01\x00")


  def grab(self): 
    
    self.send_msg(START_GET_IMAGE_TRANSFER, b'\x58\x5b\x01\x00')
    toread = 2*RAW_WIDTH*RAW_HEIGHT
    ret  = self.dev.read(0x81, 0x3F70, 1000)
    remaining = toread-len(ret)
    
    while remaining > 512:

      ret += self.dev.read(0x81, 0x3F70, 1000)
      remaining = toread-len(ret)
    status = ret[4]
    if len(ret) == RAW_HEIGHT*RAW_WIDTH*2:
      return status,np.frombuffer(ret,dtype=np.uint16).reshape(
            RAW_HEIGHT,RAW_WIDTH)
    else:
      return status,None
  
  def rescale(self, img):

    if img is None:
      return np.array([0])
    mini = img.min()
    maxi = img.max()
    return (np.clip(img-mini,0,maxi-mini)/(maxi-mini)*255.).astype(np.uint8)

  def get_image(self):

    while True:
      status,img = self.grab()

      if status == 1: 
        self.calib = self.crop(img)-1600
      elif status == 3: 
        if self.calib is not None:
          return self.correct_dead_pix(self.crop(img)-self.calib)
                
class Stream:
  
  def __init__(self, proto: SeekPro):
    self.seek = proto
    self.dev = self.seek.dev
    self.cam = None
    self.st_start = False
    self.bt_stop =False
     
  def streaming(self):
    global captured
    global stream_image
    if self.st_start is True:  
      self.cam = self.seek.get_image()
      scale = self.seek.rescale(self.cam)
      r_color = cv2.applyColorMap(scale, cv2.COLORMAP_JET)
      stream_image = r_color

      _, data = cv2.imencode(".jpg", r_color, [cv2.IMWRITE_JPEG_QUALITY, 80] )
      message = base64.encodebytes(data.tobytes())
      captured = message

  def start(self):
    if self.st_start is False:
      self.st_start = True
      if self.cam is None:
        print("Starting Camera...")
    else:
      self.streaming() 

  def stop(self):
    if self.st_start is True:
      print("Stopping Camera..")
      self.seek.stop
      print("Camera Stopped!")
     
if __name__ == "__main__": 
  ctk.set_appearance_mode("dark")
  ctk.set_default_color_theme("dark-blue")
  root = ctk.CTk()
  app = App(root)
  root.mainloop()

