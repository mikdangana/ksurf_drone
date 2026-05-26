#!/bin/sh
#SBATCH --account=def-jacobsen
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32                             # number of MPI processes
#SBATCH --time=1:00:00                                  # time (DD-HH:MM)
#SBATCH --job-name=jobkafka1
#SBATCH --mem=20G
#SBATCH --mail-user=michael.dangana@mail.utoronto.ca              # notification for job conditions
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
 

declare -A CFG=()


get_opts() {
    export START=true
    if [[ "${@}" =~ "-h" ]] || [[ "${@}" =~ "--help" ]]; then
	echo "Usage: $0 [--start|--stop|--runtest(s) n] [-h|--help]"; exit 0
    fi
    if  [[ "${@}" =~ "--stop" ]] ; then
        export START=false
    fi
    if  [[ "${@}" =~ "--addbroker" ]] ; then
	add_broker
	exit 0
    fi
    if  [[ "${@}" =~ "--runtests" ]] ; then
	export NUM_TESTS=2
        while [ $# -gt 0 ]; do
           if [[ $1 =~ "--runtests" ]]; then
             export NUM_TESTS=${2}
           fi
           shift
        done
	echo "RUNTESTS = True, NUM_TESTS = $NUM_TESTS"
	time run_tests 
	exit 0
    fi
}


get_properties() {
    echo; echo Retrieving properties...
    BASE=`dirname "${BASH_SOURCE}"`/../../
    echo BASE = $BASE
    CFG_FILE="${BASE}config/testbed.properties"
    export HOST=localhost 
    #export HOST=`hostname`
    sed -i 's/\(kafka.endpoints=\).*:/\1'$HOST':/g' $CFG_FILE
    for x in `grep = $CFG_FILE`; do
        CFG["${x//=*/}"]="${x//*=/}"
    done
    echo config file = $CFG_FILE, properties = ${!CFG[@]}
  
    KAFKA_HOME=${CFG["kafka.home"]}
    export KAFKA_HOME=`echo $KAFKA_HOME | sed 's/\(.*\/\)/\1/'`;
    #export KAFKA_HEAP_OPTS="-Xmx64m"
    #export ZK_SERVER_HEAP=512
    export CFG BASE NUM_BROKERS=${CFG["kafka.brokers"]}
    export QUEUE_REQ=${CFG["queued.max.requests"]}
    export STARTUP_TS=${CFG["kafka.startup_time_sec"]}
    export ZCFG=${KAFKA_HOME}/config/zookeeper.properties
    echo KAFKA_HOME=$KAFKA_HOME
}

install_java() {
    JRE_ARCHIVE=jre-8u351-linux.tar.gz
    JRE_URL=https://javadl.oracle.com/webapps/download/AutoDL?BundleId=247127_10e8cce67c7843478f41411b7003171c
    wget $JRE_URL -O $JRE_ARCHIVE
    $tar xvf $JRE_ARCHIVE
    export JAVA_HOME=`pwd`/jre1.8.0_351
    echo PATH=$JAVA_HOME/bin:$PATH
}


add_broker() {
    get_properties
    setup_broker $(($NUM_BROKERS))
    launch_broker $(($NUM_BROKERS))
    export NUM_BROKERS=$(($NUM_BROKERS+1))
    create_topic
    echo Added broker $((NUM_BROKERS-1))
}


setup_broker() {
     i=$1
     echo Setting up broker $i
     CFG_FILE=${KAFKA_HOME}_broker$i/config/server.properties
     cp -r $KAFKA_HOME "${KAFKA_HOME}_broker$i"
     mkdir -p /tmp/kafka-logs-$i
     sed -i 's/broker.id=0/broker.id='$(($i))'/' $CFG_FILE
     sed -i 's/#\(listeners=PLAINTEXT.*\)9092/\1'$((9092+$i))'/' $CFG_FILE
     sed -i 's/\(log.dirs=\).*/\1\/tmp\/kafka-logs-'$i'/' $CFG_FILE
     #sed -i 's/\(zookeeper.connect=.*:\).*/\1'$((2181+$i))'/' $CFG_FILE
     echo Set up broker ${KAFKA_HOME}_broker$i
}


setup_brokers() {
    echo; echo Setting up brokers...
    NUM_BROKERS=${CFG["kafka.brokers"]}
    echo NUM_BROKERS=$NUM_BROKERS, CFG_FILE=$CFG_FILE
    for i in $(seq 0 $(($NUM_BROKERS-1))); do
         setup_broker $i
    done
}


launch_broker() {
    i=$1
    echo; echo Launching broker $i...
    CFG_FILE=${KAFKA_HOME}_broker$i/config/server.properties
    sed -i 's/\(queued.max.requests*=\).*/\1'$QUEUE_REQ'/' $CFG_FILE
    ${KAFKA_HOME}_broker$i/bin/kafka-server-stop.sh 
    killpid ${BASE}broker$i.pid
    if $START; then
        sleep $STARTUP_TS
        echo "nohup ${KAFKA_HOME}_broker$i/bin/kafka-server-start.sh $CFG_FILE >broker$i.log 2>&1 &"
	    
        nohup ${KAFKA_HOME}_broker$i/bin/kafka-server-start.sh $CFG_FILE \
	    > broker$i.log 2>&1 &
        echo $! > ${BASE}broker$i.pid
        sleep $STARTUP_TS
    fi
}


create_topic() {
    topic=${CFG["kafka.topics"]//,*/}
    nohup $KAFKA_HOME/bin/kafka-topics.sh --create \
      --bootstrap-server $HOST:9092 \ #--zookeeper $HOST:2181 \
      --replication-factor $NUM_BROKERS --partitions 1 \ #$NUM_BROKERS \
      --topic $topic > topics.log 2>&1 &
    #$KAFKA_HOME/bin/kafka-topics.sh --describe --topic $topic \
     # --bootstrap-server $HOST:9092 
      #--zookeeper $HOST:2181
}


launch_brokers() {
    echo; echo Launching brokers...
    echo start = $START, startup_ts = $STARTUP_TS
    ${KAFKA_HOME}/bin/zookeeper-server-stop.sh
    killpid ${BASE}zoo.pid
    if $START; then
        sleep $STARTUP_TS
        nohup ${KAFKA_HOME}/bin/zookeeper-server-start.sh $ZCFG > zoo.log 2>&1 &
        echo $! > ${BASE}zoo.pid
    fi
    broker_logs=${CFG["kafka.broker.logdir"]//,*/}
    rm -fr ${broker_logs}
    for i in $(seq 0 $(($NUM_BROKERS-1))); do
        launch_broker $i
    done
    if $START; then
	create_topic
    fi
}

killpid() {
  PID_FILE=$1
  if [ -e $PID_FILE ]; then
    PID=`cat $PID_FILE`
    echo `ps aux | grep $PID`
    if  [[ `ps aux | grep $PID` =~ "$PID" ]] ; then
        kill -9 $PID
	echo killed pid $PID
    fi
  fi
}

launch_tracker() {
    echo; echo Launching tracker... 
    export FLASK_APP=${BASE}src/kafka/tracker.py
    killpid ${BASE}tracker.pid
    if $START; then
	rm ${BASE}tracker.log
        nohup python -m flask run >& ${BASE}tracker.log & 
        echo $! > ${BASE}tracker.pid
	sleep 20
    fi
}

launch_producers_consumers() {
    echo; echo Launching consumers...
    export NUM_PRODUCERS=1
    killpid ${BASE}consumer.pid
    for i in $(seq 0 $(($NUM_PRODUCERS-1))); do
        killpid ${BASE}producer$i.pid
    done
    if $START; then
	rm ${BASE}tracker.log
        nohup python ${BASE}src/kafka/consumer.py >& consumer.log &
	echo $! > ${BASE}consumer.pid
        sleep $STARTUP_TS
	for i in $(seq 0 $(($NUM_PRODUCERS-1))); do
            nohup python ${BASE}src/kafka/producer.py >& producer$i.log &
	    echo $! > ${BASE}producer$i.pid
        done
    fi
}


run_test() {
    setup_brokers
    launch_brokers
    #launch_tracker
    launch_producers_consumers
    if $START; then
        sleep 120
    fi
    echo; echo $0 done, output in ${CFG["data.tracker.file"]}.
}


run_tests() {
    CSV=${CFG["data.tracker.file"]}.csv
    for i in $(seq 0 $(($NUM_TESTS-1))); do
	iter=$i
	export START=false
	run_test
	sleep 60
	rm -fr /c/tmp/zoo*; rm -fr /c/tmp/kafka*; rm -fr /tmp/kafka*
	export START=true
	run_test
	sleep 420
	if [ ${iter} == 0 ]; then 
	  cp ${CSV} ${CSV}all.csv
	  echo Test results ${iter} copied
        else	
	  python ${BASE}src/kafka/tracker.py --copycolumn 2 ${CSV} ${CSV}all.csv
	  echo Test results ${iter} appended
	fi
    done
    python ${BASE}src/transformer.py --crossings ${CSV}all.csv
    echo; echo Runtests done, output in ${CSV}all.csv
}


get_properties
get_opts $@
run_test
