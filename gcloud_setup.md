# gcloud-compute-engine-tutorial
Tutorial to start and manage compute instances

---
# GCloud setup
Install gcloud
```sh
pip install gcloud
```

Initialize and authenticate your gcloud
```sh
gcloud init
gcloud auth authenticate
gcloud auth login
gcloud auth configure-docker
gcloud auth activate-service-account --key-file=[path to service account json] --project=[cloud project name]
```

In your .bashrc or .zshrc set the following 
```sh
export GOOGLE_APPLICATION_CREDENTIALS="path to service account json"
```

```sh
gcloud config list
```

---
## List of gcloud machine types

[GCloud list of machine types](https://cloud.google.com/compute/docs/machine-types)
or via the gcloud command
```sh
gcloud compute machine-types list --zone="europe-west3" --filter="name~'n1'"
```
## List of gcloud GPUs
Only N1 and A2 machine types support GPUs
[GPUs on GCloud](https://cloud.google.com/compute/docs/gpus)

## List of base deep learning images
[DeepLearning images](https://cloud.google.com/ai-platform/deep-learning-vm/docs/images)

```sh
gcloud compute images list --project deeplearning-platform-release --filter="name=pytorch-1-7" 
```

---
# Create basic compute instance with basic image
2 scripts are in the repository to create instances
To start with a base image with installed nvidia drivers
```sh
./create_gcp_instance.sh
```
From this basic image one can build a custom image. Clone your repository for example. To be able to clone and pull a private github repo follow this guide to create a [deploy key](https://gist.github.com/holmberd/dbeb8789742acfd791747772104160fe)

The config file in `.ssh/config` should look like this
```sh
Host gcloud-compute
	Hostname github.com
	User git
	IdentityFile [PATH_TO_SSH_KEY]
```
When cloning a repo the `git clone` command need to use the `Host NAME`
```sh
git clone git@gcloud-compute:rewe-digital-ri/gcloud-compute-engine-tutorial.git
```
