## Ray Cluster Commands

```sh
ray up start_ray_cluster.yaml
```

```sh
ray down start_ray_cluster.yaml
```

```sh
ray attach start_ray_cluster.yaml
```

```sh
ray exec start_ray_cluster.yaml 'cd INSTANCE_NAME && python -m RUNFILE' &
```

```sh
ray exec start_ray_cluster.yaml --port-forward=7007 'cd ray_results/EXPERIMENT_NAME && source /opt/conda/etc/profile.d/conda.sh && tensorboard --logdir=. --port=7007'
```