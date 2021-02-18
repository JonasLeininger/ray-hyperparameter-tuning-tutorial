#!/bin/bash

gcloud compute instances create "ri-gcloud-tutorial" \
                         --zone="europe-west4-a" \
                         --image-family="pytorch-latest-cu110" \
                         --image-project=deeplearning-platform-release \
                         --maintenance-policy=TERMINATE \
                         --boot-disk-size=200GB \
                         --machine-type=n1-standard-8 \
                         --accelerator="type=nvidia-tesla-p100,count=1" \
                         --metadata="install-nvidia-driver=True,proxy-mode=project_editors,container=eu.gcr.io/GLCOUD-PROJECT-NAME/gcloud-tutorial,google-logging-enabled=true" \
                         --scopes=https://www.googleapis.com/auth/cloud-platform \
                         --project=GLCOUD-PROJECT-NAME
