from flask import Flask, request
from producer import *
from time import sleep
import logging, sys, os, re
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))
from ekf import *
from utils import *

app = Flask(__name__)


def system(cmd):
    print(f"system().cmd = {cmd}")
    return os.system(cmd)


def dupe(val, n):
    return np.array([val for i in range(n)])


class Tracker:

    config = None
    is_scaled = False
    is_deterministic = False
    kf = None
    t0 = None
    count = 0

    def __init__(self, deterministic=False):
        self.config = PoissonProducer.load_configs()
        self.kf = PCAKalmanFilter(nmsmt = 2, dx = 2, normalize=True)
        self.is_deterministic = deterministic
        pickledump(self.config.get("data.tracker.file").data, 
               [["TOPIC", "PARTITION", "GLOBAL MEAN LATENCY (ns)", 
                 "GLOBAL MEAN THROUGHPUT (msg/s)", "LOCAL LATENCY (ns)"]])


    def means(self, metrics):
        size, total_latency = metrics
        mean_latency = total_latency / size
        throughput_recs = size / mean_latency
        print(f'mean_latency = {mean_latency}, throughput = {throughput_recs}')
        return [mean_latency, throughput_recs]


    def compute_latency(self, latencies, msmts, ts_ms, val):
                # compute jacobian, Hx, and normalized throughput 
                latencies.append([ts_ms - val, ts_ms-val])
                if not self.is_deterministic:
                    return self.kf.to_H(latencies, msmts)
                return latencies[-1]


    def to_metrics(self, requests, metrics, track_cadence):
        msmts, latencies, n, hj,hx = [0,0], [[0, 0]], len(requests), None, None
        ts_ms = datetime.now().timestamp()
        print(f"to_metrics.now ={ts_ms}, requests ={len(requests)}:{requests}")
        for i, consumer_record in zip(range(n), requests):
            val = float(consumer_record.value.decode("utf-8"))
            self.t0 = val if self.t0 is None else self.t0
            if is_pca():
                msmts.append(self.kf.pca_normalize(val - self.t0)) 
            else:
                msmts.append(val - self.t0)
            if self.is_deterministic or i%track_cadence < 1 or len(msmts) == 3:
                hj, hx = self.compute_latency(latencies, msmts, ts_ms, val)
            else:
                # predict latency/throughput
                latencies.append(list(self.kf.x_estimate(msmts, hj, hx))) 
            k = f"{consumer_record.topic}-{consumer_record.partition}"
            metrics[k] = [latencies[-1]+msmts[-1:]] if k not in metrics else \
                    metrics[k] + [latencies[-1]+msmts[-1:]]
        print(f"to_metrics.metrics = {metrics}")
        return metrics


    def track_latency_throughputs(self, records):
        metrics = {}
        update_rate = float(self.config.get("tracker.update.rate").data)
        mod = 0 if update_rate<=0 else 1/update_rate
        for topic_data, requests in records.items():
            print(f"topic={topic_data}, requests={len(requests)}, mod={mod}")
            #requests.sort(key=lambda rec : float(rec.value.decode("utf-8")))
            self.to_metrics(requests, metrics, mod)
        rows = []
        for k, vals in metrics.items():
            #print(f"track_l_t.vals = {vals}")
            rows += [k.split("-") + v for v in vals]
        pickleconc(self.config.get("data.tracker.file").data, rows)
        print("Tracker output for {} requests in {}".format(len(records),
              self.config.get("data.tracker.file").data))
        return rows
  

    def get_leader(self, topic, partition):
        topics_file = "topics.txt"
        kafkadir = os.path.abspath(self.config.get("kafka.home").data)
        host = re.sub(":.*", "", self.config.get("kafka.endpoints").data)
        tscript = os.path.join(kafkadir, "bin", "kafka-topics.sh")
        system(f"{tscript} --zookeeper {host}:2181" \
                  f" --describe --topic {topic} > {topics_file}")
        ldr, lines = 0, open(topics_file, 'r').readlines()
        for line in lines:
            if line.contains('Partition: ' + partition + ' '):
                ldr = re.compile(r' Leader: ([0-9,]+)').match(line).groups()[0]

        return ldr


    def scale_broker(self, topic, partition, scale_up=True):
        ldr = 0 #self.get_leader(topic, partition)     
        kafka_home = self.config.get("kafka.home").data
        sleep_ts = float(self.config.get("kafka.startup_time_sec").data)
        brokerhome = os.path.abspath(f"{kafka_home}_broker{ldr}")
        setupscript = os.path.join(os.path.dirname(__file__), "setup.sh")
        cfg_file = os.path.join(brokerhome, 'config', 'server.properties')
        cparam = "socket.receive.buffer.bytes"
        delta = 10
                                
        q_sz = int(PoissonProducer.load_configs(cfg_file).get(cparam).data)
                                    
        q_sz += delta if scale_up else (-delta if q_sz > delta else 0)
        for i in range(1):
            system(f"{setupscript} --addbroker")
                               
        self.is_scaled = scale_up
        return self.is_scaled


    @staticmethod
    def copy_column(col, src, dest):
        srcdata = np.genfromtxt(src, delimiter=",", dtype=str)
        destdata = np.genfromtxt(dest, delimiter=",", dtype=str)
        if len(destdata.shape) < 2:
            print("copy_col() malformed input {srcdata.shape} {destdata.shape}")
            return None
        v = srcdata[:, int(col)][-1]
        rows,cols = max(len(destdata), len(srcdata)), len(destdata[0])
        print(f"copy_col().srcdata.len={len(srcdata)} {len(srcdata[0])} "\
                f"dest={len(destdata)} {len(destdata[0])}, col,r={col},{rows}")
        #rowext = np.array([destdata[-1] for i in range(rows-len(destdata))])
        rowext = dupe(destdata[-1], rows-len(destdata))
        #srccol = np.concatenate((srcdata[:,int(col)],
        #                  np.array([v for i in range(rows-len(srcdata))])))
        srccol=np.concatenate((srcdata[:,int(col)], dupe(v,rows-len(srcdata))))
        print(f"copy_col().srccol = {srccol.shape}, rowext = {rowext.shape}")
        destdata = np.append(
            np.append(destdata, rowext).reshape((rows,cols)).T,
            srccol).reshape((cols+1, rows)).T
        destdata[0][-1] = f"{destdata[0][-1]} " \
                          f"{len(destdata[0]) - len(srcdata[0])}"
        print(f"copy_col().after.src = {srcdata[0:2]}, dst = {destdata[0:2]}")
        dest = dest[0:len(dest)-4]
        return savecsv(dest, destdata)


    @staticmethod
    def get_steps(cols, src):
        data = np.genfromtxt(src, delimiter=",", dtype=str).T
        if len(data.shape) < 2:
            print("get_steps() malformed input {data.shape}")
            return None
        steps = [list(map(lambda v: v[2], l))[0]
                 for l in \
                    [filter(lambda v: float(v[1])-float(v[0]) > 20,
                            zip(data[c][1:],data[c][2:],range(len(data[c]))))\
                  for c in [int(s) for s in cols]]]
        print(f"get_steps().steps = {steps}")
        return steps

    @staticmethod
    def residual_variance(cols, src, rng):
        data = np.genfromtxt(src, delimiter=",", dtype=str)
        if len(data.shape) < 2:
            print("residual_variance() malformed input {data.shape}")
            return None
        variance, residuals = [], np.array([])
        for col in cols:
            print(f"col = {col}, head = {data[1:5,int(col)]}")
            y = np.array([float(v) for v in data[rng[0]:rng[1], col]])
            x = np.array(range(len(y)))
            print(f"col = {col}, y = {y.shape}, x = {x.shape}")
            A = np.vstack([x, np.ones(len(x))]).T
            m, c  = np.linalg.lstsq(A, y, rcond=None)[0]
            residual = y - (np.multiply(m, x) + c)
            hdr = f"{data[0, col]} Residual"
            residuals = np.append(residuals.T, np.append([hdr], residual)). \
                           reshape((-1, len(residual)+1)).T
            variance.append(np.var(residual))
        savecsv(f"{src}residual", residuals)
        print(f"residual_variance().variances = {variance}")
        return variance 



    def process(self, records, with_scaling=True):
          max_latency = float(self.config.get("max_latency").data)
          min_throughput = float(self.config.get("min_throughput").data)
          for topic, partition, latency, throughput, _ in \
                    self.track_latency_throughputs(records):
              self.count += len(records)
              print(f"topic={topic}, partition={partition}, " \
                    f"latency={latency}, throughput={throughput}") 
              if latency > max_latency or throughput < min_throughput:
                if not self.is_scaled:
                    # scale up broker queue size
                    self.is_scaled = not with_scaling or \
                            self.scale_broker(topic, partition)
                    print(f"scale_up count = {self.count}")
              elif self.is_scaled:
                # scale down broker queue size
                #self.scale_broker(topic, partition, False)
                print(f"scale_down count = {self.count}")



@app.route("/")
def home():
    app.logger.info("Home visited")
    logging.info("home")
    return "<p>Hyscale KFPCA-based Kafka Tracker</p>"


@app.route("/track", methods=['POST'])
def track_requests():
    app.logger.info("Track requests = %s", request.data)
    tracker.process(request.form('records'))


class ConsumerRecord:
    key = "k"
    value = "0".encode(encoding='utf-8')
    topic = "topic1"
    partition = "partition1"
    timestamp = 0

    def __init__(self, key, value, timestamp):
        self.key = key
        self.value = value.encode(encoding='utf-8')
        self.timestamp = timestamp

    def __str__(self):
        return f"key={self.key}, value={self.value}, " \
               f"topic={self.topic}, part={self.partition}"

    def __repr__(self):
        return f"ConsumerRecord({self.key}, {self.value}, " \
                f"{self.topic}, {self.partition})"


        
if __name__ == "__main__":
  nums = lambda values : [int(v) for v in values]
  if "--copycolumn" in sys.argv:
    #python ${BASE}src/kafka/tracker.py --copycolumn 2 ${CSV} ${CSV}all.csv
    col, src, dest = sys.argv[sys.argv.index("--copycolumn")+1:]
    Tracker.copy_column(col, src, dest)
  elif "--filtercolumns" in sys.argv:
    #python ${BASE}src/kafka/tracker.py --filtercolumns 2,5 ${CSV}
    cols, src = sys.argv[sys.argv.index("--filtercolumns")+1:]
    print(f"src = {src}, cols = {cols}")
    Tracker.get_steps(cols.split(","), src)
  elif "--columnvariance" in sys.argv:
    #python ${BASE}src/kafka/tracker.py --columnvariance 2,5 ${CSV} 1:300001
    cols, src, rng = sys.argv[sys.argv.index("--columnvariance")+1:]
    print(f"src = {src}, cols = {cols}, range = {rng}")
    Tracker.residual_variance(nums(cols.split(",")), src, nums(rng.split(":")))
  else:
    tracker = Tracker()
    def get_ts(i):
        return (datetime.now()-timedelta(seconds=math.exp(10-i/20))).timestamp()
    print("tracker.process = {}".format(
        tracker.process(
            {"topic1": 
             [ConsumerRecord(f"k{i}", f"{get_ts(i)}", get_ts(i)) \
                     for i in range(20)]},
             False)))
                                      
