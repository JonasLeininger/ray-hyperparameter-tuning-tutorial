## Ray Cluster Commands
---
If you are using `poetry` the following commands need `poetry run` in front of them

---
### Local Cluster
```sh
ray start --head
```

```sh
ray stop -v
# somethimes --force is needed to kill all processes
ray stop --force
```
---
### Remote Cluster

```sh
ray up start_ray_cluster.yaml
```

```sh
ray down start_ray_cluster.yaml
```

```sh
ray attach start_ray_cluster.yaml
```

### Using Global Python Installation with pip
```sh
ray exec start_ray_cluster.yaml 'cd REPO_NAME && python -m RUNFILE' &
```
---
### Using Conda
If you use *conda* on the remote machine/cluster you need to source the `conda.sh` via
```sh
source /opt/conda/etc/profile.d/conda.sh
```
This command is needed everytime you don't ssh directly into the remote machine. Examples are *Jenkins*, *Github Actions* or *ray exec* commands. For example the following command to start tensorboard and use port forwarding:

```sh
ray exec start_ray_cluster.yaml --port-forward=7007 'cd ray_results/EXPERIMENT_NAME && source /opt/conda/etc/profile.d/conda.sh && tensorboard --logdir=. --port=7007'
```
---
### Using Poetry

Sometimes the pyenv and poetry path viables are not loaded from the `~/.bashrc` or `~/.zshrc`. Then you need to add pyenv and poetry to your PATH inside the given command.
For example inside the `start_ray_cluster_poetry.yaml`

```sh
export PATH="$HOME/.pyenv/bin:$PATH"; 
export PATH="$HOME/.poetry/bin:$PATH"; 
eval "$(pyenv init -)"; 
eval "(pyenv virtualenv-init -)"; 
pyenv local 3.8.5;
```
