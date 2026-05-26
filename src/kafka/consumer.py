from datetime import datetime
from kafka import KafkaConsumer
from kafka.structs import TopicPartition
from datetime import datetime, timedelta
from producer import PoissonProducer
from tracker import Tracker
import os, re, requests
import numpy as np
import subprocess as sp


class ActiveConsumer:

    config = None
    topic = None
    consumer = None
    tracker = None

    def __init__(self):
        self.config = PoissonProducer.load_configs()
        self.topic = self.config.get("kafka.topics").data.split(",")[0]
        self.tracker = Tracker(self.config.get("tracker.type").data=="PASSIVE")
        print(f"tracker.is_deterministic = {self.tracker.is_deterministic}")
        bootstrap_servers = self.config.get("kafka.endpoints").data.split(",")
        print(f"topic = {self.topic}, kafka.srv = {bootstrap_servers}")
        self.consumer = KafkaConsumer(
            self.topic, bootstrap_servers=bootstrap_servers)
        print(f"Init done")


    def get_latency_throughputs(self, records):
        metrics = {}
        ts = datetime.now() - timedelta(minutes=60)
        ts_ms = ts.timestamp()*1000.0
        for topic_data, consumer_records in records.items():
            for consumer_record in consumer_records:
                print(f"Consumer_record = {consumer_record}")
                k = f"{consumer_record.topic()}-{consumer_record.partition()}"
                latency = ts_ms - float(consumer_record.value.decode("utf-8"))
                metrics[k] = np.array([1, latency]) if k not in metrics else \
                             np.array([1, latency]) + metrics[k]
        def means(metric):
            [size, total_latency] = metric
            mean_latency = total_latency / size
            throughput_recs = size / mean_latency
            print(f'mean_latency = {latency}, throughput = {throughput_recs}')
            return mean_latency, throughput_recs
        tuples = map(lambda i: i[0].split("-") + means(i[1]), metrics.items())
        return list(tuples)


    def setup_listener(self):
        # Timestamp to look for events
        # Look for events created from 60 minutes ago
        ts = datetime.now() - timedelta(minutes=60)
        # Convert to epoch milliseconds
        ts_milliseconds = ts.timestamp()*1000.0

        print(f'Looking for offsets of : {ts} ({ts_milliseconds})')

        # We only retrieve from partition 0 for this example
        # as there is only one partition created for the topic
        # To find out all partitions, partitions_for_topic can be used.
        topic_partition_0 = TopicPartition(self.topic, 0)
        timestamps = {topic_partition_0: ts_milliseconds}
        offsets = self.consumer.offsets_for_times(timestamps)
        offset_p0 = offsets[topic_partition_0]
        offset = 0 if offset_p0 is None else offset_p0.offset
        print(f"offsets = {offsets}, offset_p0 = {offset_p0}")

        self.consumer.seek(partition=topic_partition_0, offset=offset)


    def listen(self):
        self.setup_listener()
        self.max_latency = float(self.config.get("max_latency").data)
        self.min_throughput = float(self.config.get("min_throughput").data)
        hostname = re.sub(":.*", "", self.config.get("kafka.endpoints").data)
        print(f"max_latency = {self.max_latency}, " \
              f"throughput = {self.min_throughput}, host = {hostname}")
        while True:
            print('polling...')
            records = self.consumer.poll(timeout_ms=1000)
            self.tracker.process(records)
            #resp = requests.post(f"http://{hostname}:5000/track", data=records)
            #print(f'posted to tracker response = {resp.text}')


if __name__ == "__main__":
    consumer = ActiveConsumer()
    consumer.listen()
