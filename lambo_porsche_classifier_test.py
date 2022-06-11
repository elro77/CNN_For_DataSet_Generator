import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mig
import os
from PIL import Image


#training on GPU

gpu_=tf.test.gpu_device_name()
tf.device(gpu_)

PATH = 'C:\\Yolo_DataSet_Generator\\YoloDatasetGenerator\\bin\\Debug\\netcoreapp3.1\\Generated Images 56\\ImageDatasets'
testPath = "C:\\Users\\Elroye\\Pictures\\car tests"
classes = os.listdir(PATH)




#load saved model
new_model = tf.keras.models.load_model('C:\\Yolo_DataSet_Generator\\YoloDatasetGenerator\\bin\\Debug\\netcoreapp3.1\\Generated Images 56\\Model\\my_model')


c1_data_path=[os.path.join(testPath, img) for img in os.listdir(testPath) if img.endswith(".jpg")]
len(c1_data_path)


for i in range(0,len(c1_data_path)):
  img = Image.open(str(c1_data_path[i]))
  img = img.resize((256, 256), Image.NEAREST)
  img = np.asarray(img).astype('uint8')

  img = img.reshape((256,256,3))
  arrayOfImage=np.zeros((1,256,256,3))
  arrayOfImage[0] = img /255.0  
  pred_ = new_model.predict(arrayOfImage)
  
  plt.imshow(img)
  title_ = 'Predict:' + str(classes[pred_.argmax()] + " Accuracy: " + str(pred_[0][pred_.argmax()]))
  plt.title(title_,size=11)
  plt.show()


