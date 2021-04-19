import pandas as pd
import numpy as np
from sklearn.svm import SVR, SVC
import matplotlib.pyplot  as plt
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
import librosa
import os
import scipy.fftpack as sf
import pathlib
import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
import pickle
from google.colab import drive
drive.mount('/content/drive')

def load_data(datasetPath, csvFileName):
  '''
    Parameters:
           dataset path, should be like /content/gdrive/My Drive/...
           csvFileName; name of csv file
           The csv file should have two columns, age and audioFilenames
    Returns: 
          pandas dataframe
  '''

  from google.colab import drive
  drive.mount('/content/drive') 
  
  return pd.read_csv("{}/{}".format(datasetPath, csvFileName))

def feat_extract(path,dataset):
  '''
    Parameters:
            path, like .gdrive/My Drive/wavedata
            dataset, the dataset containing audio filenames and age labels

    Creates dataset.csv file containing features and age labels

    Returns:
            pandas dataframe with features and age labels
  '''
  header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
  
  for i in range(1, 21): 
    header += f' mfcc{i}'
  header += ' label'
  header = header.split()

  file = open('dataset.csv', 'w', newline='')
  with file:
    writer = csv.writer(file)
    writer.writerow(header)

  df = dataset

  for (i,filename) in enumerate(df.filename):
    breathAudio = f'{path}/{filename}'
  
    y, sr = librosa.load(breathAudio, mono=True, duration=15)
    rmse = librosa.feature.rmse(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #'''
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
      to_append += f' {np.mean(e)}'
    '''
    to_append = f'{filename} {chroma_stft} {rmse} {spec_cent} {spec_bw} {rolloff} {zcr}'

    #shape = mfcc.shapea = np.zeros(shape)
    for e in mfcc:
      to_append += f' {e}'
    '''

    file = open('dataset.csv', 'a', newline='')
    with file:
      writer = csv.writer(file)
      writer.writerow(to_append.split())
  try:
    data = pd.read_csv('dataset.csv')
    for i in range(len(data['label'])):
      data["label"][i] = df["label"][i]
  except KeyError:
    pass
  data.to_csv('dataset.csv')
  return data

def scale_and_split(dataset):
  '''
  Scales features based on mean/std of the training data.

  Parameters:
  - dataset: a pandas dataframe that contains features and age labels
  
  Returns
  - X_train, X_test, y_train, y_test
  '''
  
  #Run this only when we have new data
  #'''
  
  dataset = dataset.drop(['filename'],axis=1)
  X = np.array(dataset.iloc[:, :-1], dtype = float)
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)

  np.save("/content/drive/My Drive/Virufy/breath/data/all/mean.npy", mean)
  np.save("/content/drive/My Drive/Virufy/breath/data/all/std.npy", std)

  y = dataset.iloc[:, -1]
  lb = LabelBinarizer()
  y = lb.fit_transform(y)
  
  X = (np.array(dataset.iloc[:, :-1], dtype = float) - mean.reshape(1,26) ) /std.reshape(1,26)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state = 20, shuffle = True)

  return  X_train, X_test, y_train, y_test

def scale_and_split_test(dataset):
  '''
  Scales features based on mean/std of the training data.

  Parameters:
  - dataset: a pandas dataframe that contains features and age labels
  
  Returns
  - X, y
  '''
  mean = np.load("/content/drive/My Drive/Virufy/breath/data/all/mean.npy")
  std = np.load("/content/drive/My Drive/Virufy/breath/data/all/std.npy")

  dataset = dataset.drop(['filename'],axis=1)

  X = np.array(dataset)
  X = (np.array(dataset) - mean.reshape(1,26) ) /std.reshape(1,26)

  return  X

def train(X_train, y_train):
  classifier = SVC(kernel = 'rbf', probability = True)
  classifier.fit(X_train, y_train)
  return classifier

def get_score(X_train, X_test, y_train, y_test, model):
  '''
    Returns train_score, test_score
  '''
  scores = model.score(X_train, y_train)
  print ("Train score" , scores)
  scores2 = model.score(X_test, y_test)
  print ("Test score" , scores2)
  return [scores, scores2]

def save_model(model):
  '''
    Saves model into .sav file
    Returns None
  '''
  filename = 'breathing-model.sav'
  pickle.dump(model, open(filename, 'wb'))

def predict(audio_fp, model_fp):
  
  model = pickle.load(open(model_fp, 'rb'))
  
  audio_path, audio_name = audio_fp.rsplit('/',1)
  df = pd.DataFrame ({'filename':  [audio_name]}, columns = ['filename'])
  df = feat_extract(audio_path, df)
  df.drop(['label'], axis = 1, inplace = True)
  pd.set_option("display.max_rows", None, "display.max_columns", None)

  X = scale_and_split_test(df)

  pred = model.predict_proba(X)
  return pred

if __name__ == "__main__":

  # double check filemane and dataset path after receiving data
  DATASETPATH = '/content/drive/My Drive/Virufy/breath/data/all'
  CSV_FILENAME = 'dataset.csv'

  df = load_data(DATASETPATH, CSV_FILENAME)
  df = feat_extract(DATASETPATH,df)
  X_train, X_test, y_train, y_test = scale_and_split(df)

  model = train(X_train, y_train)
  save_model(model) #saved in existing directory

  get_score(X_train, X_test, y_train, y_test, model)