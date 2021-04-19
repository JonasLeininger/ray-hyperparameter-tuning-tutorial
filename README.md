# Ray Hyperparameter Tuning Tutorial

This repo is an introduction to ray and a cluster setup in the GCP

## Poetry
This tutorial uses **Poetry** to manage the python libraries. Go to the official [Poetry](https://python-poetry.org/) page for detailed information. I wrote down a quick start for poetry in [POETRY.md](https://github.com/JonasLeininger/ray-hyperparameter-tuning-tutorial/blob/main/POETRY.md)

## Google Cloud
Using GCP look up the scripts in `workflow_scripts` and [GcloudSetup.md](https://github.com/JonasLeininger/ray-hyperparameter-tuning-tutorial/blob/main/GcloudSetup.md)

## Hyperparameter Tuning Overview
[Here](https://github.com/JonasLeininger/ray-hyperparameter-tuning-tutorial/blob/main/HyperparameterTuning.md) is a general overview of Hyperparameter Tuning techniques with algorithms and schedules.

## Hyperparameter Tuning with Ray and Ray[tune]
Read about a ray specific commands and how to start a cluster [here](https://github.com/JonasLeininger/ray-hyperparameter-tuning-tutorial/blob/main/RayClusterCommands.md). Example tune python implementations, using the class API, for the [Trainer](https://github.com/JonasLeininger/ray-hyperparameter-tuning-tutorial/blob/main/src/hyperparam_tune/training/trainer.py) and an experiment [runner](https://github.com/JonasLeininger/ray-hyperparameter-tuning-tutorial/blob/main/src/hyperparam_tune/training/tune.py) 