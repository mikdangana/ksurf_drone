from config import *
#from lstm import Lstm
from utils import *
#from plot import *
import config as cfg
import math, numpy as np
import yaml, logging, logging.handlers
#import matplotlib.pyplot as plt
#import tensorflow as tf
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter, CubatureKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
from math import sqrt
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract, eye, reshape
from numpy.linalg import inv
from functools import reduce
from random import random
from datetime import *
from transformer import get_layer, get_model, prepare_data, get_histories_in_order


logger = logging.getLogger("Kalman_Filter")


tconfig = load_testbed_config()
ttype = tconfig.get("tracker.type").data 

# KF types = ["UKF", "EKF", "EKF-PCA", "KF", "AKF-PCA"]
kf_type = "EKF" if ttype=="PASSIVE" else ttype
vb = False


def is_ekf():
    return kf_type.startswith("EKF") or kf_type.startswith("AKF")


def is_pca():
    return kf_type.endswith("PCA")


def is_akf():
    return kf_type.startswith("AKF")


def is_pca_or_akf():
    return is_pca() or is_akf()


# Test the accuracy of an EKF using the provide measurement data
def ekf_accuracy(ekf, msmt, indices=None, label="", predict=True, host=None):
    return ekf_accuracies(ekf, msmt, indices, label, predict, host)[-1]


# Test the accuracies of an EKF per measurement 
def ekf_accuracies(ekf, msmt, indices=None, label="", predict=True, host=None):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    _ = ekf_predict(ekf) if predict else None
    ids = cfg.find(get_config("lqn-hosts"), host)
    state = [list(t.values())[0] for t in tasks[ids[0]:ids[-1]+1]]
    (state, n_state) = (array(state), len(ids))
    logger.info("state = " + str(state) + ", msmt = " + str(len(msmt)))
    [accuracies, mean, state_ns, prior] = mean_accuracy(ekfs[0], indices, msmt)
    (max_mean, means) = ([mean, 0], [mean])
    for i in range(len(ekfs[1:])):
        _, mean, _, _ = mean_accuracy(ekfs[i+1], indices, msmt)
        max_mean = [mean, i+1] if mean > max_mean[0] else max_mean
        means.append(mean)
    swap(ekfs, max_mean[1])
    mean = max_mean[0]
    logger.info(label + " x_prior = " + str(shape(get_ekf(ekfs).x_prior)) + 
        ", zip(prior,state,accuracies) = " + 
        str(list(zip(prior, state_ns, accuracies))) + 
        ", means = " + str(means) + ", accuracy = " + str(mean))
    return [[state_ns, accuracies, prior], mean]


def ekf_predict(ekf):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    for ekf in ekfs:
        get_ekf(ekf).predict()
    return get_ekf(ekf).x_prior


def get_ekf(ekf):
    while isinstance(ekf, list) or isinstance(ekf, tuple):
        ekf = ekf[0]
    if kf_type.startswith("UKF"):
        return ekf if isinstance(ekf, UnscentedKalmanFilter) else ekf['ekf']
    elif kf_type == "KF":
        return ekf if isinstance(ekf, KalmanFilter) else ekf['ekf']
    else:
        return ekf if isinstance(ekf, ExtendedKalmanFilter) else ekf['ekf']


def mean_accuracy(ekf, indices, state):
    (ekf,(m,c)) =(ekf['ekf'],ekf['mc']) if isinstance(ekf,dict) else (ekf,(1,0))
    nums = lambda ns : [n[0] for n in ns]
    prior = lambda kf: nums(m * get_ekf(kf).x_prior + c)
    acc = lambda pt: 1 - abs(pt[1] - pt[0]) 
                         #/abs(pt[0]+1e-9) #norms[pt[2] % len(norms)]
    accuracies = [acc(p) for p in zip(state, prior(ekf), range(len(state)))]
    logger.info("accuracies = " + str(accuracies) + \
        ", state = " + str(state) + ", prior = " + str(prior(ekf)))
    mean = avg([accuracies[i] for i in indices] if indices else accuracies) 
    return [accuracies, mean, state, prior(ekf)]


def swap(lst, i):
    tmp = lst[0]
    lst[0] = lst[i]
    lst[i] = tmp


def read2d(coeffs, width, start, end):
    vals = array(coeffs[start:end])
    vals.resize(width, width)
    return vals



# Build and update an EKF using the provided measurement data
def build_ekf(coeffs, z_data, linear_consts=None, nmsmt = n_msmt, dx =dimx, \
              hx=None): 
    global n_msmt
    global dim
    (dimx, n_msmt) = (dx, nmsmt)
    logger.info(f"build_ekf().dimx = {dimx}, dx = {dx}, n_msmt = {n_msmt}")
    if kf_type.startswith("UKF"):
        ekf = build_unscented_ekf(hx)
    elif kf_type == "KF":
        ekf = KalmanFilter(dim_x=4,dim_z=2)
        ekf.x = eye(4)
        #ekf = KalmanFilter(dim_x = dimx, dim_z = n_msmt)
        #(ekf.P, ekf.R, ekf.Q) = (eye(dimx), eye(dimx)*.1, eye(n_msmt)*.1)
    else:
        ekf = ExtendedKalmanFilter(dim_x = dimx, dim_z = n_msmt)
    if len(coeffs):
        r = update_ekf_coeffs(ekf, coeffs)
        return update_ekf(ekf, z_data, r, linear_consts)
    return update_ekf(ekf, z_data)


def build_unscented_ekf(hx=None):
        #pts = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
        pts = JulierSigmaPoints(2)
        def fx(x, dt):
            # state transition function - predict next state based
            # on constant velocity model x = vt + x_0
            F = np.array([[1, dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]], dtype=float)
            F = np.array([[1, 0], 
                          [0, 1]], dtype=float)
            return np.dot(F, x)
   
        def h(x):
            # measurement function - convert state into a measurement
            # where measurements are [x_pos, y_pos]
            #return np.array([x[0], x[2]])
            return np.array([x[0], x[1]]) if hx is None else hx(x)
   
        #ekf =UnscentedKalmanFilter(dim_x=4,dim_z=2,dt=.1,hx=h,fx=fx,points=pts)
        ekf = UnscentedKalmanFilter(dim_x=2,dim_z=2,dt=.1,hx=h,fx=fx,points=pts)
        return ekf


def update_ekf_coeffs(ekf, coeffs):
        coeffs = array(coeffs).flatten()
        if n_coeff == dimx * 2 + n_msmt:
            ekf.Q = symmetric(array(coeffs[0:dimx]))
            ekf.F = symmetric(array(coeffs[dimx:dimx*2]))
            r = symmetric(array(coeffs[-n_msmt:]))
        else:
            ekf.Q = read2d(coeffs, dimx, 0, dimx*dimx)
            ekf.F = read2d(coeffs, dimx, dimx*dimx, dimx*dimx*2)
            r = read2d(coeffs, n_msmt, -n_msmt*n_msmt, n_coeff)
        logger.info("ekf.Q={}, F = {}, r = {}".format(ekf.Q, ekf.F, r))
        return r


def update_ekf(ekf, z_data, R=None, m_c = None, Hj=None, H=None):
    logger.info("z_data = " + str((array(z_data).shape)))
    (ekfs, start) = (ekf if isinstance(ekf, list) else [ekf], datetime.now())
    priors = [[] for i in ekfs]
    for i,z in zip(range(len(z_data)), z_data):
        z = reshape(z, (array(z).size, 1))
        logger.info("z = " + str((len(z), z.size, array(z).shape)))
        def h(x):
            return H(x) if H else m_c[0]*x if m_c else x
        def hjacobian(x):
            m = m_c[0] if m_c else 1
            return Hj(x) if Hj else m * identity(len(x)) 
        update_ekf_msmt(z, ekfs, priors, R, hjacobian, h)
    logger.info("priors,z_data,ekfs = " + str((array(priors).shape, 
                                               array(z_data).shape,len(ekfs))))
    return (ekf, priors)


def update_ekf_msmt(z, ekfs, priors, R, hjacobian, h):
        for j,ekf in zip(range(len(ekfs)), ekfs):
            ekf = get_ekf(ekf)
            ekf.predict()
            priors[j].append(ekf.x_prior)
            if kf_type.startswith("UKF"):
                z = [z[0][0], z[1][0]]
                ekf.update(z, R=R if len(shape(R)) else ekf.R, hx=h)
            elif kf_type == "KF":
                z = [z[0][0], z[1][0]]
                ekf.update(z)
            else:
                ekf.update(z, hjacobian, h, R if len(shape(R)) else ekf.R)



def norm_bias(a, bias=0.0001):
        #a=a/np.min(a) # flip -ve values
        a = a - 1.01*np.min(a)
        a1=a*1/0.01
        b=a1 #(a1 - (1-0.89)*np.min(a1));
        c=b/np.linalg.norm(b) #(1 if np.max(b)==np.min(b) else np.max(b)-np.min(b))
        d=c/np.max(c)
        d=d/np.max(d)
        print(f"a = {a}, b = {b}, min(b) = {np.min(b)}, c = {c}, d = {d}")
        #exit(0)
        return d



def pad(values, N):
    return np.array(list(np.zeros(N-len(values)) if N>len(values) else []) + \
            list(values[-N:]))



class PCAKalmanFilter:

    ekf = None
    kf = None
    msmts, x_hist = [], []
    n_components = None
    model = None
    normalize = False
    update_count = 0
    px, py, Jprev = None, None, np.eye(2)
    hx, hj = None, None

    def __init__(self, nmsmt=None, dx=None, n_components=10, H=None, 
                       normalize=False, att_fname=None, att_col=None):
        self.ekf = build_ekf([], [], nmsmt = nmsmt, dx = dx, hx = H)[0]
        self.n_components = n_components
        self.msmts = [0 for i in range(self.n_components)]
        train_fname = sys.argv[sys.argv.index("-ft")+1] if "-ft" in sys.argv else att_fname
        train_col = sys.argv[sys.argv.index("-z")+1] if "-z" in sys.argv else att_col
        data = (prepare_data(False, fname=att_fname, col=train_col))
        self.model = get_model(data=data)
        self.normalize = normalize
        self.kf = self


    #
    # Computes prediction Jacobian
    # Params: y - predicted variable, x - prediction variable, w - window
    #
    def to_jacobian(self, y, x, w=1):
        print(f"to_jacobian().y={y}, x={x}")
        i2 = -2-w if 1+w > len(x) else -1-w
        dy = np.subtract(np.array([y[-1],y[-1]]), np.array([y[-2], y[-2]]))
        dx = np.subtract(np.array(x[-1-w]), np.array(x[i2]))
        J = self.Jprev if x[-1-w] == x[i2] else \
            np.array([[dy[r]/dx[c] for c in [0,1]] for r in [0,1]]) 
        self.Jprev = J
        print(f"to_jacobian.y={y}, x={x}, dy={dy}, dx={dx}, J={J}")
        return J


    def Hj(self, y, x):
        J = self.to_jacobian(y, x)
        return lambda x1: J

    
    def Hx(self, y, x):
        J = self.to_jacobian(y, x)
        x = np.array(x).T
        def H(x1):
            #x1 = x1.T[0]
            dx = np.subtract(np.array(x1).T, np.array(x[-1])).T
            Jx = np.matmul(J, np.array(dx))
            Hx = np.add(np.array(y[-1:]), Jx.T)
            #print(f"H().x={x}, x1={x1}, y={y}, Jx={Jx}, dx={dx}, Hx = {Hx}")
            return Hx
        return H


    def pair(self, items, prev, defval=0):
        return [defval if prev is None else prev, items[-1]]


    def to_H(self, x_hist, msmts):
                    self.kf.ekf.x = np.array([[v] for v in x_hist[-1]]) \
                            if is_ekf() else np.array(x_hist[-1]) # UKF
                    x = self.pair(x_hist, self.px, [0,0])
                    y = self.pair(msmts, self.py) 
                    self.px = x_hist[-1]
                    self.py = msmts[-1]
                    hj, hx = self.Hj(y, x), self.Hx(y, x)
                    #print("MD: x={}, y={}, H(x)={}".format(x,y,hx(x)))
                    #exit(0)
                    if kf_type=="UKF" and self.update_count == 1:
                        self.kf = PCAKalmanFilter(nmsmt = 2, dx = 2, H=hx)
                        self.kf.ekf.x = np.array(x_hist[-1])
                    self.kf.update(msmts[-2:], Hj=hj, H=hx)
                    self.update_count += 1
                    self.hj, self.hx = hj, hx
                    return hj, hx


    def x_estimate(self, msmts, hj=None, hx=None):
        (hj, hx) = (self.hj, self.hx) if hj==None else (hj, hx)
        # predict latency/throughput
        x = self.kf.predict([msmts[-2],msmts[-1]],Hj=hj,H=hx)[-1][-1].T
        print(f"x_estimate().msmts = {msmts}, prior = {x}")
        return x[0] if is_ekf() else x # UKF



    def update(self, *args, **kwargs):
        _, priors = update_ekf(self.ekf, args, **kwargs)
        return priors


    def predict(self, *args, **kwargs):
        _, priors = update_ekf(self.ekf, args, **kwargs)
        return priors



    def pca_attention(self, values, n, vb=False, alpha=1):
        print(f"pca_attention().values.0 = {values}, n = {n}") if vb else None
        inputs = get_histories_in_order(values)
        print(f"pca_attention().values.1 = {inputs[-n:]}") if vb else None
        values=np.array(get_layer(self.model,inputs[-n:].reshape(n,n,1),5,vb))
        print(f"pca_attention().values.2 = {values.T}") if vb else None
        weights = sum(values) #norm_bias(values.T[-1])[-1] 
        weights = weights / max(weights) # deleted
        weights = np.nan_to_num(weights, nan=1.0)
        print(f"pca_attention().weights = {weights.T}") if vb else None
        values = weights.T * inputs #weights * inputs
        print(f"pca_attention().values.3 = {values}, shape={inputs.shape}")\
                if vb else None
        #values = values.reshape((values.shape[0]*values.shape[1]))
        print(f"pca_attention().values.4 = {values}, shape={inputs.shape}") if vb else None
        return values

    
    def pca_normalize(self, msmt, is_scalar=True, pca_th=0.01):
        n, N = self.n_components, self.n_components**2
        self.msmts.append(msmt)
        values=[self.msmts[0] for i in range(N-len(self.msmts))]+self.msmts
        if kf_type.startswith("AKF"): # Attention
            values, N1 = self.pca_attention(values, n, vb=vb), n
        if is_pca():                  # PCA
            mu = np.mean(pad(values, N).flatten(), axis=0)
            print(f"normalize().n = {n}, vals = {values.shape}, N = {pad(values, N).shape}") if vb else None
            pca = getpca_raw(n, pad(values, N).reshape(n, n)) 
            print(f"normalize().pca = {pca}, evecs={pca.components_}, " \
              f"evar={pca.explained_variance_}, mu={mu}, "
              f"valuesN={np.array(values[-N:]).reshape(n, n)}") if vb else None
            scores = pca.transform(np.array(values[-N:]).reshape(n, n))
            #print(f"normalize().scores = {scores}, msmts = {self.msmts}")
            pca_components = [np.zeros((1,n))[0] if v<pca_th else c for v,c in \
                zip(pca.explained_variance_, pca.components_)] # noise reduction
            print(f"normalize().comps ={pca_components}, scores={scores[:,:n]}") if vb else None
            msmt_hat = np.dot(scores[:,:n], pca_components) + mu 
            msmt_hat = self.norma(msmt_hat.flatten())
        else:                         # Normalize
            msmt_hat = values[-N:]
        res = msmt_hat[-1][-1] if is_scalar else np.array(msmt_hat).T
        print(f"normalize().msmts = {self.msmts[-N:]}," \
              f"msmt_hat = {msmt_hat}, res = {res}") if vb else None
        return res


    def norma(self, v, force=False):
        norm = np.linalg.norm(v) 
        return v / ( norm if norm>0 else 1 ) if is_pca() or force else v



def ekf_track(coeffs, z_data):
    ekf, points = build_ekf(coeffs, z_data)
    return list(zip(z_data, points[0]))


def plot_ekf():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    logger.info("plot_ekf() done")


def test_coeffs(generators):
    if n_coeff == dimx * 2 + n_msmt:
        return concatenate([ones(dimx), ones(dimx), ones(n_msmt)])
    else:
        return flatlist(ones((dimx,dimx))) + flatlist(identity(dimx)) + \
               flatlist(ones((n_msmt,n_msmt)))


def get_generators():
    generators = [lambda x: 0 if x<50 else 1, #math.pow(1.01, x), 
                  math.exp, 
                  math.sin, #lambda x: random(),
                  math.erf, #lambda x: math.sin((x-10)/10) + random()*0.25, 
                  math.erf,
                  lambda x: 1 if int(x/50) % 2==1 else 0]
    return generators



def estimate_weights():
  ekf, ecount = build_ekf([], [], nmsmt=n_coeff, dx=n_coeff), 0
  def estimator(zs, ws, bs, X, Y, xval, yval):
    global ecount
    logger.info("estimate_w.z,w,b,xv,yv="+str((zs,ws,bs,xval.shape,yval.shape)))
    x, y, ecount = ws[-1], zs[-1], ecount + 1
    if x==None or y == None:
        logger.info("Nil x,y = " + str((x,y)))
        return ws[-1], bs[-1]
    logger.info("estimate_w.y, x, yval = " + str((y, x, bs[-1], yval.T)))
    x, y1 = tf.reshape(x, y.shape), y
    xt = tf.transpose(x)
    with tf.Session() as sess:
        yval = np.reshape(yval, Y.shape)
        sess.run(tf.compat.v1.global_variables_initializer())
        x = sess.run(y1, feed_dict = {X: xval, Y: yval})
        logger.info("ncost = x.xt = " + str((x, np.matmul(x, x.T))))
        try:
            return weights_from_ekf(x, y1, bs, ekf)
        except:
          logger.error("Error while computing H, probably x.xT not invertible")
          logger.info("returning ws,bs = " + str((ws[-1], bs[-1])))
          return ws[-1], bs[-1]
  return estimator



def weights_from_ekf(x, y1, bs, ekf):
        if None==ekf.H or ecount % Hf == 0:
            ekf.H = np.matmul(np.matmul(y1.eval(), x.T),
                              np.linalg.inv(np.matmul(x, x.T)))
        H = ekf.H
        #H =tf.matmul(tf.matmul(y1, xt), tf.linalg.inv(tf.matmul(x, xt))).eval()
        logger.info("estimate_w.H = " + str((H.shape, x.shape, 
                    x.size, n_entries, n_lstm_out, n_msmt, n_coeff, H)))
        Hx = lambda xi: [logger.info("Hx.xi = "+str((H.shape, 
            xi.reshape((n_entries, int(dimx/n_entries))).shape, xi.shape))), 
            np.matmul(H, xi.reshape((n_entries, int(dimx/n_entries)))).
               reshape(dimx, 1)][1]
        logger.info("estimate_w.z_data = " + str(([zs[-1].eval()][0].shape)))
        _, w = update_ekf(ekf, [x], R=None, m_c = None, Hj=None, H=Hx)
        logger.info("estimate_w.w = " + str((w)))
        return w, bs[-1]



def test_lstm_ekf():
    tf.enable_eager_execution()
    #ws, bs = grad_fn(x, ws, bs, Hj)
    Lstm().tune_model(15, grad_fn=estimate_weights())



# Testbed to unit test EKF using hand-crafted data
def test_ekf(generate_coeffs = test_coeffs):
    generators = get_generators()
    m_cs = [(10.0, 0.0) for i in range(len(generators))]
    (coeffs, accuracies, predictions) = (generate_coeffs(generators), [], [])
    for n in range(len(generators)):
        (train, test) = test_zdata(generators, n)
        logger.info("train[0:1] = " + str(train[0:1]))
        ekf, _ = build_ekf(coeffs, train, m_cs[n])
        ekf = {"ekf": ekf, "mc": m_cs[n]}
        accuracies.append(avg([ekf_accuracy(ekf, t) for t in test])) 
        logger.info("train=" + str(len(train)) + ", test = " + str(len(test)) + 
            ", accuracy = " + str(accuracies[-1]) + ", fn = " + str(n) + 
            " of " + str(len(generators)))
        predictions.append(ekf_track(coeffs, concatenate([train, test])))
        pickledump("predictions" + str(n) + ".pickle", predictions[-1])
    logger.info("accuracies = " + str(accuracies))
    return predictions



def test_zdata(generators, n):
    z_data = []
    g = generators[n]
    for v in (array(range(300))/30 if g in [math.exp,math.sin] else range(100)):
        msmt = [g(v) for m in range(n_msmt)]
        z_data.append(msmt)
    split = int(0.75 * len(z_data))
    return (z_data[0 : split], z_data[split : ])



def predict(ekf, msmts, dy=2):
    msmts=np.reshape(msmts.flatten()[-dy:],(1,dy))
    priors = update_ekf(ekf.ekf if is_pca_or_akf() else ekf, msmts)[1]
    priors = np.array([priors[-1][-1][-1]]) if is_pca() else \
               to_size(priors[-1], msmts.shape[1], msmts.shape[0])
    print(f"predfn.priors = {priors}, msmts = {msmts}") if vb else None
    return priors



def test_pca():
    f = os.path.join(sys.path[0],'..','data','mackey_glass_time_series.csv')
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
    pca = sys.argv[sys.argv.index("-pc")+1] if "-pc" in sys.argv else "false"
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'P'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'P'
    priors, ekf = [], PCAKalmanFilter(nmsmt=2, dx=2, normalize=True, 
                                      att_fname=f, att_col=xcol) \
          if is_pca_or_akf() else build_ekf([], [], nmsmt=2, dx=2) 
    def predfn(msmts, x_hist = None): 
        print(f"predfn.x_hist = {x_hist}, msmts = {msmts}, type = {kf_type}") if vb else None
        if is_pca_or_akf():
            msmts = msmts[-1] if depth(msmts)<3 else msmts[-1][-1]
            x_hist = [msmts[:-1]]
            msmts = ekf.pca_normalize(msmts[-1], is_scalar=False)
            if is_pca():
              ekf.to_H([[x,x] for x in x_hist] if np.array(x_hist).shape==(1,) \
                       else [[x[0],x[0]] for x in x_hist], msmts)
        return predict(ekf, msmts, dy=2)
    tag = f"_{kf_type}"
    err = test_pca_csv(f,xcol,ycol,None,predfn,dopca=pca.lower()=="true",
                 pre=tag, predictions=priors)
    return err #priors



# Supported parameters: 
# python ekf.py [-t EKF|AKF|UKF] [-f file] [-ft file] [-x col] [-y col] [-z col] [--testpcacsv] [--testlstm] [--testekf]
#
if __name__ == "__main__":
    kf_type = sys.argv[sys.argv.index("-t")+1] if "-t" in sys.argv else kf_type
    if "--testpcacsv" in sys.argv:
        test_pca()
    elif "--testlstm" in sys.argv:
        test_lstm_ekf()
    elif "--testekf" in sys.argv:
        test_ekf()
    else:
        test_pca()
    print("Output in lstm_ekf.log")
