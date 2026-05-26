/root/kafka-env/bin/python3 -c "from kafka import *; import logging; print(KafkaProducer(bootstrap_servers='127.0.0.1:9092').send('test-topic', bytes('100', 'utf-8')))"
