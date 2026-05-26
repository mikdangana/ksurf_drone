# Derived from https://www.kaggle.com/code/michaeldangana/pressure-prediction-using-deep-learning/edit
import tensorflow as tf
import pandas as pd
import numpy as np
import os, random, time
import matplotlib.pyplot as plt
from datetime import datetime

#import tensorflow_text


def get_features():
    dataset = os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"file = {dataset}")
    train_dataset = pd.read_csv(os.path.join(dataset, "twitter_trace.csv"))
    print(f"head = {train_dataset.head()}")

    print(datetime.strptime(train_dataset['End'][0], "%m/%d/%Y %H:%M"))
    train_dataset['Start'] = train_dataset['Start'].transform(
        lambda d: time.mktime(datetime.strptime(d, "%m/%d/%Y %H:%M").timetuple()))
    train_dataset['End'] = train_dataset['End'].transform(
        lambda d: time.mktime(datetime.strptime(d, "%m/%d/%Y %H:%M").timetuple()))
    train_dataset['Tweet Count'] = train_dataset['Tweet Count'].transform(
        lambda c: int(c))
    print(f"end = {np.unique(train_dataset['End'])}")
    print(f"start = {np.unique(train_dataset['Start'])}")
    print(train_dataset.head())

    max_len, w = len(train_dataset) - 4, 96
    features = train_dataset.drop(['Start'], axis=1)
    target = train_dataset.drop(['Start','End'], axis=1)
    features = np.array(features)[0:max_len]
    print(features.shape)
    features = features.reshape(int(features.shape[0]/w), w, features.shape[1])    
    target = np.array(target)[2:max_len+2].reshape(features.shape[0], w)
    print(f"features = {features.shape}, target = {target.shape}")
    return features, target, w


def build_model(features, target, w):
    norm = tf.keras.layers.Normalization(input_shape = [w, features.shape[2],], axis = -1)
    norm.adapt(features)
    m = tf.keras.Sequential([
        norm,
        tf.keras.layers.Conv1D(128, 3, activation = "relu"),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(256, 3, activation = "relu"),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(w,)  
    ])
    m.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss= "mae")
    print(m.summary())
  
    history = m.fit(features, target, validation_split = 0.2, epochs = 3500, batch_size = 512,
                callbacks = [tf.keras.callbacks.EarlyStopping(patience = 260, 
                                                              monitor = 'val_loss', 
                                                              mode = 'min', 
                                                              restore_best_weights=True)])
    return history, m


def predict(m, features, target):
    m.evaluate(features, target)
    y_pred = m.predict(features, batch_size = 512)
    print(f"tgt = {target[0]}, y_pred = {y_pred[0]}")
    draw_result(0, 50, target, y_pred)
    draw_result(110, 160, target, y_pred)
    draw_result(160, 220, target, y_pred)
    plt.show()
    

def draw_result(start, end, target, y_pred):
    plt.figure(figsize = (20, 7))
    plt.plot(np.reshape(target[start:end], -1), linewidth=5, label = "actual values")
    plt.plot(np.reshape(y_pred[start:end], -1), linewidth=2, label = "predict values")
    plt.legend()


def plot(history):
    #print(f"loss = {history.history['loss']}")
    #print(f"val_loss = {history.history['val_loss']}")
    plt.figure(figsize = (20, 5))
    plt.plot(history.history['loss'], label = "loss")
    plt.plot(history.history['val_loss'], label = "val_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("RNN")
    feat, target, w = get_features()
    train_history, m = build_model(feat, target, w)
    plot(train_history)
    predict(m, feat, target)



