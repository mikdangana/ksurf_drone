from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from scipy.signal import savgol_filter
from functools import reduce
from ekf import test_pca
from utils import pickledump
from keras import backend as k
import keras
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import sys



class GridLSTM(LSTM):

  def __init__(self, *args, **kwargs):
      if not len(args):
          args = (32) # units = 32
      print("args = {}, kwargs = {}".format(args, kwargs))
      super(GridLSTM, self).__init__(*args, **kwargs)
      self.args, self.kwargs = args, kwargs


  def build(self, input_shape):
      print("build.input_shape = {}".format(input_shape))
      self.lstms = [Bidirectional(LSTM(*self.args, **self.kwargs)) \
                    for i in range(input_shape[1])]
      (a,b,c) = input_shape
      list(map(lambda lstm: lstm.build((a,1,c)), self.lstms))


  def build_basic(self, input_shape):
      self.w = [self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True) for i in range(input_shape[1])]
      self.b = [self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True) for i in range(input_shape[1])]
      print("input_shape = {}, = {}, b = {}".format(
          input_shape, self.w[0].shape, self.b[0].shape))
      #super(GridLsTM, self).build(input_shape)


  def call(self, inputs):
      #print("inputs = {}, w = {}".format(inputs.shape, self.w[0].shape))
      print("inputs = {}".format(inputs[:,0:1,:].shape))
      return reduce(tf.add, map(lambda i: self.lstms[i].call(inputs[:,i:i+1,:]),
                                range(inputs.shape[1])))

  def call_basic(self, inputs):
      out = [tf.matmul(inputs[:,i,:], self.w[i]) + self.b[i] \
              for i in range(inputs.shape[1])]
      return reduce(tf.add, out)



def my_train_test_split(x, y, test_size=0.2):
    mark = int((1-test_size)*len(x)) if test_size*len(x) <= 2000 else 2000
    #return x[0:mark,:], x[mark:,:], y[0:mark,:], y[mark:,:]
    #return x[mark:,:], x[0:mark,:], y[mark:,:], y[0:mark,:]
    X_train,X_test,y_train,y_test = \
            x[0:mark,:],x[mark:len(x)-1,:],y[1:mark+1],y[mark+1:]
    return X_train, X_test, y_train, y_test



def to_grid(x, w=50):
    flt = lambda n: 0 if math.isnan(float(n)) else float(n)
    x = np.array([list(x[i:i+w])+list(np.zeros((max(0,i+w-len(x)),1))) \
                  for i in range(len(x))])
    x = np.array([[flt(x[i,j]) for j in range(w)] for i in range(len(x))])
    return x


def generate_from_csv(fname, xcol = '$uAppP', ycol = '$fGet_n', y1col = '$fGet',
                      predfn = None, dopca = True, pre = "", 
                      filter_savgol=False, filter_kalman=False, w=50):
    (r, rows, hdrs) = (-1, {}, [])
    flt = lambda n: 0 if math.isnan(float(n)) else float(n)
    rows = pd.read_csv(fname)
    y = np.array([flt(v) for v in rows[ycol][w:]] + \
                    [0 for i in range(w)])
    x = np.array(to_grid(rows[xcol], w=w))
    #x, y = x[0:1000], y[0:1000] 
    #print("xcol = {}", rows[xcol][0:5])
    if filter_savgol:
        x = savgol_filter(x, 5, 2)
        x = np.array([[max(0,v) for v in r] for r in x])
        print("savgol.x = {}".format(x))
    elif filter_kalman:
        x = to_grid(np.array(test_pca()))
        print("kalman.x = {}".format(x))
    
    X_train, X_test, y_train, y_test = my_train_test_split(x, y, test_size=0.8)
    print("X_train = {}, y_train = {}".format(X_train[0:5], y_train[:5]))
    print("b4.X_train = {}, y_train = {}".format(X_train.shape, y_train.shape))
    print("X_test = {}, y_test = {}, filter = {}".format(
            X_test.shape, y_test.shape, label(filter_savgol, filter_kalman)))
    #y_train = pd.get_dummies(y_train[:,1]).values
    #y_test = pd.get_dummies(y_test[:,1]).values
    #print("aft.X_train = {}, y_train = {}".format(X_train.shape, y_train.shape))

    #return X_train, y_train[:,1], X_test, y_test[:,1]
    return X_train, y_train[:], X_test, y_test[:]



def generate_data(test_split = 0.2, variable=False, filter_savgol=False, w=50):
    print("---Generating Data---")

    X = []
    y = []

    for i in range(10000):
        x_hold = []

        if variable:
            value = np.random.randint(2, size=np.random.randint(1, w))
        else:
            value = np.random.randint(2, size=w)

        for x in value:
            x_hold.append(int(x))
        if sum(x_hold) % 2 == 1:
            y.append(0)
        else:
            y.append(1)
        X.append(x_hold)

    X = np.array(X)
    print("X = {}".format(X))
    if filter_savgol:
        X = savgol_filter(X, 5, 2)
    print("X1 = {}".format(X.shape))
    #exit(0)
    y = np.array(y)

    X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.8)
    #y_train = pd.get_dummies(y_train).values
    #y_test = pd.get_dummies(y_test).values

    return X_train, y_train, X_test, y_test



def weighted_mse(y_true, y_pred, alpha=1):
    return alpha * k.mean(k.square(y_true - y_pred))


def create_GridLSTM_model(rows=10000, length=50):
    print('---Creating GridLSTM model---')
    embed_dim = 128
    lstm_out = 200

    model = Sequential()
    model.add(Embedding(rows, embed_dim, input_length=length)) #, dropout=0.2))
    model.add(GridLSTM(lstm_out)) #, dropout_U = 0.2, dropout_W = 0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mse', #weighted_mse, # 'mse',
                  optimizer='adam',
                  metrics=['mean_squared_error'])
                  #metrics=['accuracy'])
    print(model.summary())
    return model



def label(sf, kf, **kwargs):
    if 'filter_savgol' in kwargs:
        sf = sf if sf else kwargs['filter_savgol']
    if 'filter_kalman' in kwargs:
        kf = kf if kf else kwargs['filter_kalman']
    return "Savitsky-Gorlay" if sf else "Kalman" if kf else "GridLSTM"



def run_test(*args, **kwargs):
    X_train,y_train,X_test,y_test = generate_data(w=50) if not len(args) else \
                                       generate_from_csv(*args, **kwargs, w=50)
    resfile = "grid_results.pickle"    
    print("x_train, x_test = {}, {}".format(len(X_train), len(X_test)))
    print("x_test, y_test = {}, {}".format(X_test[0:10], y_test[0:10]))

    model = create_GridLSTM_model(len(X_train), len(X_train[0]))
    epochs = int(sys.argv[sys.argv.index("-e")+1]) if "-e" in sys.argv else 5
    model.fit(X_train, y_train, batch_size=10, epochs=epochs)
    if "--eval" in sys.argv:
        err, acc = model.evaluate(X_test, y_test, batch_size=4)
        print("Model Eval: err, acc = {}, {}".format(err, acc))
        return err, acc
    predictions = np.array(
        [model.predict(X_test[i:i+1])[-1][0] for i in range(len(X_test))])
    errs = np.array([abs(p[0]-p[1]) for p in zip(y_test, predictions)])
    results = np.array(list(zip(y_test,predictions,errs)))
    print("epochs,y,pred,err= {}".format(epochs, results))
    pickledump(resfile, results)
    print("output in {}.csv, err = {}, std = {}".format(resfile, np.mean(errs), np.std(errs)))
    return np.mean(errs), np.mean(errs)



if __name__ == "__main__":
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else None
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'P'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'P'
    sf = True if "--savgol" in sys.argv else False
    kf = True if "--kalman" in sys.argv else False
    def test(sf=sf, kf=kf):
        return run_test(filter_savgol=sf, filter_kalman=kf) if f is None else \
            run_test(f, xcol=xcol, ycol=ycol, filter_savgol=sf,filter_kalman=kf)
    res = []
    if "-a" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n")+1]) if "-n" in sys.argv else 1
        for sf, kf in [(True, False), (False, True), (False, False)]:
            res.append([("",label(sf, kf)) + test(sf, kf) for i in range(n)])
    elif "-n" in sys.argv:
        res = [[test() for i in range(int(sys.argv[sys.argv.index("-n")+1]))]]
    else:
        test()
    for entry in res:
        [print("err, acc = {}, {}".format(err, acc)) for err, acc in entry]
        #print("")




