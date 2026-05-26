#!/bin/bash

DIR=${0//provision_stack.sh/}
if [ -z $DIR ]; then DIR="./"; fi
echo DIR = $DIR, 0 = $0

# Source the OpenStack RC file for authentication
source ${DIR}rrg-PROJECT_OWNER-openrc.sh


create_instance() {
    # Define variables
    INSTANCE_NAME=$1 #"my-compute-instance"
    FLAVOR="c1-7.5gb-36"       # c1-7.5gb-36, c4-15gb-144 
    if [ $# -gt 1 ]; then FLAVOR=$2; fi
    IMAGE="Ubuntu-24.04-Noble-x64-2024-06"      # Name of the image to use (ensure it exists in your project)
    NETWORK="rrg-PROJECT_OWNER-network" # Name of the network to attach
    KEYPAIR="<KEYPAIR_NAME>"      # Name of the keypair for SSH access
    SECURITY_GROUP="<SECURITY_GROUP>"  # Name of the security group (default is commonly used)
    CINIT="cinit_worker.yaml"       # c1-7.5gb-36, c4-15gb-144 
    if [ $# -gt 2 ]; then CINIT=$3; fi
    
    EXISTS=`openstack server show "$INSTANCE_NAME" 2>&1`
    echo EXISTS = $EXISTS
    if [ ! -z "${EXISTS}" ]; then
	echo; echo Deleting $INSTANCE_NAME
        openstack server delete "$INSTANCE_NAME" --wait
    fi

    echo; echo Creating instance $INSTANCE_NAME...

    # Create the instance
    openstack server create \
      --flavor "$FLAVOR" \
      --image "$IMAGE" \
      --network "$NETWORK" \
      --security-group "$SECURITY_GROUP" \
      --key-name "$KEYPAIR" \
      --user-data "$CINIT" \
      "$INSTANCE_NAME"

}

create_producers() {
    for i in $(seq 2 1 5); do
      create_instance "ksurf-producer$i" "c4-15gb-144" "cinit_producer_new.yaml"
    done
}

create_stack() {
    #create_instance "ksurf-master" "c4-15gb-144" "cinit_master.yaml"
    MASTER_IP=`openstack server show ksurf-master | grep addresses | awk '{print $4}' | sed 's/.*=//'`
    echo MASTER_IP = $MASTER_IP
    sed -i 's/192.168.[0-9]*.[0-9]*/'${MASTER_IP}'/g' $DIR/cinit_producer_new.yaml
    sed -i 's/192.168.[0-9]*.[0-9]*/'${MASTER_IP}'/g' $DIR/cinit_worker.yaml
    #create_instance "ksurf-producer" "c4-15gb-144" "cinit_producer_new.yaml"
    for i in $(seq 3 1 5); do
	echo; echo creating instance ksurf-worker${i}...
        create_instance "ksurf-worker${i}" "c1-7.5gb-36" "cinit_worker.yaml"
    done
}


delete_stack() {
    openstack server delete "ksurf-master" --wait
    openstack server delete "ksurf-producer" --wait
    for i in $(seq 1 1 1); do
        openstack server delete "ksurf-worker${i}" --wait
    done
}


create_stack
#create_producers

# Check instance status
echo; echo Listing openstack servers...
openstack server list | grep ksurf



