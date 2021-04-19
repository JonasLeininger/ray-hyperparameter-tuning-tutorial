#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit
fi

SOURCE_VM=$1 
IMAGE_NAME=$2 
FAMILY=$3 
GCP_PROJECT=[YOUR_GCP_PROJECT]
# delete image if it already exists
if [[ "$IMAGE_NAME" == $(gcloud compute images list --format="value(NAME)" --filter="name=(${IMAGE_NAME})") ]]; then
   echo "delete image $IMAGE_NAME"
   gcloud compute images delete "$IMAGE_NAME" -q
fi

# create image
gcloud compute images create ${IMAGE_NAME} --project=${GCP_PROJECT} --family=${FAMILY} --source-disk=${SOURCE_VM} --source-disk-zone=europe-west4-a --storage-location=eu