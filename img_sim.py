import os
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping



#Extracting image paths
train_files = os.listdir('train\output')
test_files = os.listdir('test')

print("Number of Training Images:",len(train_files))
print("Number of Test Images: ",len(test_files))
train_files = pd.DataFrame(train_files,columns=['filepath'])
test_files = pd.DataFrame(test_files,columns=['filepath'])

# #converting into .csv file for future reference.
# train_files.to_csv('train_file.csv')
# test_files.to_csv('test_file.csv')
# print(test_files)

def image2array(file_array,floder):
 """
 Reading and Converting images into numpy array by taking path of images.
 Arguments:
 file_array - (list) - list of file(path) names
 Returns:
 A numpy array of images. (np.ndarray)
 """
 image_array = []
 for path in tqdm(file_array['filepath']):
    if path != 'desktop.ini':
      img = cv2.imread(floder+'\\'+path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (224,224))
      image_array.append(np.array(img))
 image_array = np.array(image_array)
 image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
 image_array = image_array.astype('float32')
 image_array /= 255
 return np.array(image_array)

train_data = image2array(train_files,"train\output")
print("Length of training dataset:",train_data.shape)
test_data = image2array(test_files,"test")
print("Length of test dataset:",test_data.shape)

def encoder_decoder_model():

  """
  Used to build Convolutional Autoencoder model architecture to get compressed image data which is easier to process.
  Returns:
  Auto encoder model
  """
  #Encoder
  model = Sequential(name='Convolutional_AutoEncoder_Model')
  model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 3),padding='same', name='Encoding_Conv2D_1'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_1'))
  model.add(Conv2D(128, kernel_size=(3, 3),strides=1,kernel_regularizer = tf.keras.regularizers.L2(0.001),activation='relu',padding='same', name='Encoding_Conv2D_2'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_2'))
  model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',kernel_regularizer= tf.keras.regularizers.L2(0.001), padding='same', name='Encoding_Conv2D_3'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_3'))
  model.add(Conv2D(512, kernel_size=(3, 3), activation='relu',kernel_regularizer= tf.keras.regularizers.L2(0.001), padding='same', name='Encoding_Conv2D_4'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2,padding='valid', name='Encoding_MaxPooling2D_4'))
  model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Encoding_Conv2D_5'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

  #Decoder
  model.add(Conv2D(512, kernel_size=(3, 3), kernel_regularizer = tf.keras.regularizers.L2(0.001),activation='relu', padding='same', name='Decoding_Conv2D_1'))
  model.add(UpSampling2D((2, 2), name='Decoding_Upsamping2D_1'))
  model.add(Conv2D(512, kernel_size=(3, 3), kernel_regularizer = tf.keras.regularizers.L2(0.001), activation='relu', padding='same', name='Decoding_Conv2D_2'))
  model.add(UpSampling2D((2, 2), name='Decoding_Upsamping2D_2'))
  model.add(Conv2D(256, kernel_size=(3, 3), kernel_regularizer = tf.keras.regularizers.L2(0.001), activation='relu', padding='same',name='Decoding_Conv2D_3'))
  model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_3'))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.001), padding='same',name='Decoding_Conv2D_4'))
  model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_4'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.001), padding='same',name='Decoding_Conv2D_5'))
  model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_5'))
  model.add(Conv2D(3, kernel_size=(3, 3), padding='same',activation='sigmoid',name='Decoding_Output'))
  return model

model = encoder_decoder_model()
model.summary()

optimizer = Adam(learning_rate=0.001)
model = encoder_decoder_model()
model.compile(optimizer=optimizer, loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=6,min_delta=0.0001)
checkpoint = ModelCheckpoint('encoder_model.h5', monitor='val_loss', mode='min', save_best_only=True)
history = model.fit(train_data, train_data, epochs=35, batch_size=32,validation_data=(test_data,test_data),callbacks=[early_stopping,checkpoint])

# Plot training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# def feature_extraction(model, data, layer = 14):

#     """
#     Creating a function to run the initial layers of the encoder model. (to get feature extraction from any layer of the model)
#     Arguments:
#     model - (Auto encoder model) - Trained model
#     data - (np.ndarray) - list of images to get feature extraction from trained model
#     layer - (int) - from which layer to take the features(by default = 4)
#     Returns:
#     pooled_array - (np.ndarray) - array of extracted features of given images
#     """

#     encoded = K.function([model.layers[0].input],[model.layers[layer].output])
#     encoded_array = encoded([data])[0]
#     pooled_array = encoded_array.max(axis=-1)
#     return encoded_array

# encoded = feature_extraction(model,train_data[:10],9)

# knn = KNeighborsClassifier(n_neighbors=9,algorithm='ball_tree',n_jobs=-1)
# knn.fit(np.array(data),np.array(labels))

# def predictions(label,N=8,isurl=False):

#     """
#     Making predictions for the query images and returns N similar images from the dataset.
#     We can either pass filename or the url for the image.
#     Arguments:
#     label - (string) - file name of the query image.
#     N - (int) - Number of images to be returned
#     isurl - (string) - if query image is from google is set to True else False(By default = False)
#     """

#     if isurl:
#         img = io.imread(label)
#         img = cv2.resize(img,(224,224))
#     else:
#         img_path = '/content/dataset/'+label
#         img = image.load_img(img_path, target_size=(224,224))
#     img_data = image.img_to_array(img)
#     img_data = np.expand_dims(img_data,axis=0)
#     img_data = preprocess_input(img_data)
#     feature = model.predict(img_data)
#     feature = np.array(feature).flatten().reshape(1,-1)
#     res = knn.kneighbors(feature.reshape(1,-1),return_distance=True,n_neighbors=N)
#     results_(img,list(res[1][0])[1:])