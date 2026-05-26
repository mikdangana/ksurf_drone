from datetime import datetime
from jproperties import Properties
from kafka import KafkaProducer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta
from scipy.special import factorial
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformer import get_dataset
import numpy as np


class PoissonProducer:

    config = None
    num_points = 0.0
    num_test_msg = 30000
    # Topic name
    topic = ""
    producer = None


    def __init__(self):
        self.config = self.load_configs()
        self.num_points = float(self.config.get("poisson.n_per_t").data)
        self.topic = self.config.get("kafka.topics").data.split(",")[0]
        if '--topic' in sys.argv:
            self.topic = sys.argv[sys.argv.index('--topic')+1]
        # Bootstrap servers - you can input multiple ones with comma seperated.
        # for example, bs1:9092, bs2:9092
        bootstrap_servers = self.config.get("kafka.endpoints").data.split(",")
        if '--brokers' in sys.argv:
          bootstrap_servers=sys.argv[sys.argv.index('--brokers')+1].split(",")
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers[0])


    @staticmethod
    def load_configs(path=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')),
            'config', 'testbed.properties')):
        config = Properties()
        with open(path, 'rb') as config_file:
            config.load(config_file)
        return config


    def poisson_next_ms(self, t):
        plambda = float(self.config.get("poisson.lambda").data)
        n = np.arange(0, self.num_points)
        prob = np.power(plambda*t,n)/factorial(n)*np.exp(-plambda*t)
        #print(f"prob.argmax = {prob.argmax(axis=0)}")
        return prob.argmax(axis=0)


    def send(self, t, msg_bytes):
        ts = self.poisson_next_ms(self.num_test_msg-t)/self.num_points
        print(f'Sending message {msg_bytes.decode("UTF-8")} after {ts} seconds')
        #time.sleep(ts)
        future = self.producer.send(self.topic, msg_bytes)
        result = future.get(timeout=60)
        self.producer.flush()


    def send_all(self):
        for t in range(self.num_test_msg):
            ts = datetime.now() - timedelta(minutes=60)
            # Convert to epoch milliseconds
            ts_ms = ts.timestamp() #*1000.0
            self.send(t, bytes(f'{ts_ms}', 'utf-8'))


class TwitterProducer:

    config = None
    num_test_msg = 3
    bucket = 0
    # Topic name
    topic = ""
    producer = None
    bkts = None


    def __init__(self):
        data = get_dataset()
        self.config = self.load_configs()
        self.num_test_msg = sum(data['Tweet Count'])
        self.bkts = np.concatenate(data['Tweet Count'].transform(
                lambda d: np.array([d for i in range(d)])))
        self.topic = self.config.get("kafka.topics").data.split(",")[0]
        if '--topic' in sys.argv:
            self.topic = sys.argv[sys.argv.index('--topic')+1]
        # Bootstrap servers - you can input multiple ones with comma seperated.
        # for example, bs1:9092, bs2:9092
        bootstrap_servers = self.config.get("kafka.endpoints").data.split(",")
        if '--brokers' in sys.argv:
          bootstrap_servers=sys.argv[sys.argv.index('--brokers')+1].split(",")
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers[0])


    @staticmethod
    def load_configs(path=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')),
            'config', 'testbed.properties')):
        config = Properties()
        with open(path, 'rb') as config_file:
            config.load(config_file)
        return config


    def tweet_delay_ms(self, t):
        bkt_size = self.bkts[t]
        #if not bkt_size == self.bkts[t-1 if t>0 else 0]: # new bucket
        #    time.sleep((60/max(1, bkt_size))/(1000**3))
        #granularity=1min, with 1s=1ms for simulation, see src/twitter_search.py
        return 60 / max(1, bkt_size)


    def send(self, t, msg_bytes):
        ts = self.tweet_delay_ms(t)
        #print(f'Sending message {msg_bytes.decode("UTF-8")} after {ts} secs')
        future = self.producer.send(self.topic, msg_bytes)
        result = future.get(timeout=60)
        self.producer.flush()


    def send_all(self):
        print(f'Twitter.send_all().num_msg = {self.num_test_msg}')
        for t in range(self.num_test_msg):
            #self.send(t, bytes(f'tweet-{t}', 'utf-8'))
            ts_ms = (datetime.now() - timedelta(minutes=60)).timestamp() #*1000
            # Convert to epoch milliseconds
            self.send(t, bytes(f'{ts_ms}', 'utf-8'))


if __name__ == "__main__":

    poisson = PoissonProducer()
    workload = poisson.config.get("workload.type").data
    print(f"Producer.workload = {workload}")
    if "--poisson" in sys.argv or "POISSON" == workload:
        poisson.send_all()
    elif "--twitter" in sys.argv or "TWITTER" == workload:
        TwitterProducer().send_all()
    else:
        poisson.send_all()

