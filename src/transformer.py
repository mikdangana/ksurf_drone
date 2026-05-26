# Derived from https://keras.io/examples/timeseries/timeseries_transformer_classification/
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
import os, random, sys, time
import pandas as pd
import plotly.express as px
import keras
from keras import layers
from timeit import timeit
from datetime import datetime
from kf_attention import KfAttention


def show_includes(list_files=False):
    for p in sys.path:
        print(f"{p}")
        if list_files and os.path.isdir(p):
            os.listdir(p)
        #print(f"files = {os.listdir(p) if os.path.isdir(p) else None}")



def get_dataset(fname=None, vb=False):
    f = fname if fname else "twitter_trace.csv"
    datadir = os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"file = {datadir}") if vb else None
    train_dataset = pd.read_csv(fname if fname and os.path.exists(fname) else \
                                os.path.join(datadir, f))
    print(f"head = {train_dataset.head()}") if vb else None

    d0 = datetime.strptime("01/01/2023 00:00", "%m/%d/%Y %H:%M");
    if not 'Start' in train_dataset or not 'End' in train_dataset:
        return train_dataset
    print(datetime.strptime(train_dataset['End'][0], "%m/%d/%Y %H:%M")) if vb else None
    train_dataset['Start'] = train_dataset['Start'].transform(
        lambda d: (datetime.strptime(d, "%m/%d/%Y %H:%M") - d0).total_seconds())
    train_dataset['End'] = train_dataset['End'].transform(
        lambda d: (datetime.strptime(d, "%m/%d/%Y %H:%M") - d0).total_seconds())
    train_dataset['Tweet Count'] = train_dataset['Tweet Count'].transform(
        lambda c: int(c))
    return train_dataset


def flatten_counts(train_dataset):
    if not 'Tweet Count' in train_dataset:
        return train_dataset
    # 'Tweets' is a set of durations between tweets
    tweets = train_dataset['Tweet Count'].transform(
        lambda c: np.array([1e-6 if i>0 else 60/(c*1000**1) for i in range(c)]))
    dataset = pd.DataFrame(data={'Tweets': np.concatenate(tweets)})
    counts = train_dataset['Tweet Count'].transform(
        lambda c: np.array([c for i in range(c)]))
    dataset['Tweet Count'] = np.concatenate(counts)
    print(f"flatten_counts().head = {dataset.head()}")
    return dataset


def get_crossings(filename):
    print(f"file = {filename}")
    train_dataset = pd.read_csv(filename)
    print(f"head = {train_dataset.head()}")
    dataset = []
    base = "GLOBAL MEAN LATENCY (ns)" 
    for i in range(10):
        col = base if i==0 else f"{base} {i}"
        if not col in train_dataset.columns:
            continue
        vals = train_dataset[col]
        vals = np.array(vals.transform(lambda v: 
            float(str(v).split()[0].replace("[", "").replace("]", ""))))
        vals = list(filter(lambda i: i[1]-vals[i[0]-1]>30, 
                           list(zip(range(len(vals)), vals))[1:]))
        print(f"vals = {vals}")
        dataset.append(list(map(lambda t: t[0], vals))[0])
    dataset = np.array(dataset)
    print(f"p75-p25 = {np.percentile(dataset, 75)-np.percentile(dataset, 25)}")
    return dataset


def get_histories_in_order(a, max_age=10):
    b = list(np.zeros(max_age**2-len(a) if max_age**2>len(a) else 0))
    b.extend(a)
    return np.array(b[-max_age**2:]).reshape(max_age, max_age)


def get_histories(feature, max_age=10):
    hists = []
    for age in range(max_age):
        pad = np.array([feature[0] for i in range(age)])
        hists.append(np.concatenate((pad, feature[0:len(feature)-age])))
    return np.array(hists).T


def get_history(feature, start):
    pad = feature[start:].transform(lambda d: feature[0])
    return pd.concat((pad, feature[0:start]))


def get_features(verbose=True, fname=None, col=None):
    col = 'Tweet Count' if col is None else col
    train_dataset = flatten_counts(get_dataset(fname=fname))
    train_dataset[col] = pd.to_numeric(train_dataset[col], errors="coerce").fillna(0)
    max_len, w, h = len(train_dataset), 25, -10
    features = train_dataset[[col]]
    f, (f0, f1) = 10, features.shape #7*features.shape[1], features.shape
    print(f"f, f0, f1, shape.1 = {(f, f0, f1, 1)}") if verbose else None
    for i in range(f - features.shape[1]):
        features[f"h{i}"] = get_history(train_dataset[col], -f-i)
    #features['nodeid'] = features[col] # TODO: hack, find a better solution
    #features['timestamp'] = features[col] # TODO: hack, find a better solution
    print(f"features = {features[0:10]}, cols = {features.columns}") if verbose else None
    target = train_dataset[col]
    features = np.array(features) 
    target = np.array(target)
    features = features.reshape((features.shape[0], features.shape[1], 1))
    target = target.reshape((target.shape[0], 1, 1))
    N = int(0.8*len(features))
    if verbose:
        print(f"features = {features.shape}, target = {target.shape}")
    return features[0:N], target[0:N], features[N:], target[N:], f1


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def prepare_data(verbose=False, fname=None, col=None):
    x_train, y_train, x_test, y_test, w = get_features(verbose, fname, col)

    # Step 1: Flatten both to 1D
    y_train_flat = y_train.reshape(-1)
    y_test_flat = y_test.reshape(-1)

    # Step 2: Concatenate and compute number of unique labels
    n_classes = int(np.max(np.concatenate((y_train_flat, y_test_flat)))) + 1

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    #y_train[y_train == -1] = 0
    #y_test[y_test == -1] = 0
    return x_train, y_train, x_test, y_test, n_classes



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = KfAttention(
        #key_dim=head_size, num_heads=num_heads, dropout=dropout
        head_size, num_heads, dropout=dropout
    )(x, training=x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    print(f"build_model().x = {x}, input_shape = {input_shape}")
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    print(f"build_model().pool.x = {x}")
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def pred_loss(x_pred, targets):
    """Compute the Kalman filter loss using predicted Q and R."""
    loss = torch.zeros(1, requires_grad=True, dtype=torch.float32)  # ✅ Ensure loss tracks gradients

    error = (x_pred - targets) ** 2  # Squared error
    loss = loss + error  # ✅ Accumulate loss while maintaining gradients

    return loss  # ✅ Keep it inside computation graph


def train_eval(x_train, y_train, x_test, y_test, n_classes):
  input_shape = x_train.shape[1:] #x_train[0:x_train.shape[1]].shape 
  print(f"train_eval().input_shape={input_shape}, xtrain.shape={x_train.shape}")

  # $N=4$ layers, each with dimension $d_{model}=256$, feed-forward $d_{ff}=4$, attention heads $h=4, d_{k}=d_{v}=256$ and dropout rate of 0.25
  model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[16384], #128],
    mlp_dropout=0.4,
    dropout=0.25,
    n_classes=n_classes,
  )

  model.compile(
    loss="mse", #pred_loss, #"sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mse"] #"sparse_categorical_accuracy"],
  )
  model.summary()

  callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

  print(f"xtrain = {x_train[0:10]}, ytrain = {y_train[0:10]}")
  history = model.fit(
    x_train,
    y_train,
    validation_split=0.4,
    epochs=300, #300,
    batch_size=64,
    callbacks=callbacks
  )

  #plot(history)
  model.evaluate(x_test, y_test, verbose=1)
  save_model(model)

  #return get_layer(model, x_test[0:20], 2)

  return model


def save_model(model):
    path = os.path.join(os.path.dirname(__file__), "..","data","kfmodel.keras")
    return model.save(path)


def get_model(data = None):
    path = os.path.join(os.path.dirname(__file__), "..","data","kfmodel.keras")
    if os.path.exists(path):
        return keras.models.load_model(path, compile=False)
    else:
        return train_eval(*data)


def get_layer(model, inputs, depth=3, verbose=False):
    print(f"get_layer().inputs = {inputs.shape}") if verbose else None
    print(f"get_layer().layers = {len(model.layers)}") if verbose else None
    value, n_layers = inputs, 3
    for i in range(depth): 
        print(f"get_layer().layer = {model.layers[i]}") if verbose else None
        try:
            value = model.layers[i](value)
        except:
            value = model.layers[i](value, value) 
        #  if isinstance(model.layers[i], KfAttention) or \
        #     isinstance(model.layers[i], layers.core.tf_op_layer.TFOpLambda) else \
        if verbose and i == depth - 1:
          print(f"get_layer(): layer {i} = {value.shape}, " \
                f"type={type(model.layers[i])}, value.T={tf.transpose(value)}")
    return value



def plot_attention(attention_val, inputs):
    (h, w, _), a = attention_val.shape, np.array(attention_val)
    a = a.reshape((h, w))
    print(f"w = {w}, h = {h}, att_val.shape = {attention_val.shape}")
    #attention_df = np.array(attention_val).reshape((2*h, int(w/2)))
    df = pd.DataFrame(np.concatenate((a,a)),columns=[f"h{i}" for i in range(w)])
    attention_val = np.array(attention_val)
    attention = [attention_val[:,i][0][-1] if i<w else 0 for i in range(h)]
    attention = np.flip(np.array(attention), axis=0)
    ids, counts = np.array(range(h)), np.array(inputs[:,-1])
    print(f"counts = {counts.shape}, att = {np.array(attention).shape}")
    df['Time (minutes)'] = np.concatenate((ids, ids))
    df['Tweet Count'] = np.concatenate((counts[:,0], attention))
    df['Distribution'] = np.concatenate((counts[:,0], attention))
    #df['Series'] = np.concatenate(([0 for i in ids], attention))
    #df = px.data.iris()
    print(f"df = {df}")
    #print(f"iris = {px.data.iris()}")
    #fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
    #                 size='petal_length', hover_data=['petal_width'])
    fig = px.scatter(df,x="Time (minutes)",y="Tweet Count",color="Distribution",
                     size='Time (minutes)', hover_data=['Time (minutes)'])
    fig.show()


def plot(history):
    print(f"history = {history.history}")
    #print(f"loss = {history.history['loss']}")
    #print(f"val_loss = {history.history['val_loss'_]}")
    plt.figure(figsize = (20, 5))
    plt.plot(history.history['loss'], label = "loss")
    plt.plot(history.history['val_loss'], label = "val_loss")
    plt.legend()
    plt.show()



def get_arg_val(*opts):
    for arg in opts:
        if arg in sys.argv:
            return sys.argv[sys.argv.index(arg) + 1]
    return None



if __name__ == "__main__":
  if "-h" in sys.argv or "--help" in sys.argv:
      print(f"Usage: python {__file__} [--crossings csv|--li|--list-includes]")
  elif "--crossings" in sys.argv:
      print(get_crossings(get_arg_val("--crossings")))
  elif "-li" in sys.argv or "--list-includes" in sys.argv:
      show_includes()
  elif "-a" in sys.argv or "--accuracy" in sys.argv:
    dataset_file = get_arg_val("-a", "--accuracy")
    data = (prepare_data(True))
    t = timeit(lambda: get_layer(get_model(data=data),data[2][0:20],10,False), 
               number=1)
    print(f"layer.time = {t} s")
  elif "--hist" in sys.argv:
    print(f"histories = {get_histories(range(20))}")
  elif "--train" in sys.argv:
    if "-f" in sys.argv:
        dataset_file = get_arg_val("-f", "--file")
    x_train, y_train, x_test, y_test, n_classes = prepare_data()
    model = train_eval(x_train, y_train, x_test, y_test, n_classes)
  else:
    print("Transformer")
    #x_train, y_train, x_test, y_test, n_classes = prepare_data()
    #model = train_eval(x_train, y_train, x_test, y_test, n_classes)
    data = (prepare_data(True))
    model = get_model(data=data)
    #attention_val = np.array([
    #    np.array(get_layer(model, data[2][i:i+20], 3, True)[:,0,0]) \
    #    for i in range(20)]).T
    #print(f"attention_val.shape = {attention_val.shape}")
    #plot_attention(attention_val.reshape((20,20,1)), data)
    #print(f"inputs = {data[2][0:20]}")
    t = timeit(lambda: get_layer(model,data[2][0:20],10,False), 
               number=100)
    print(f"layer.time = {t} s")
    #plot_attention(get_layer(model, data[2][0:20], 5, True), data[2][0:20])

