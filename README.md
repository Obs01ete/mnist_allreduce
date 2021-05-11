# Tutorial to implement different versions of all-reduce

## Create and activate conda environment
1. Install Anaconda python virtual environment manager (miniconda is recommended)
2. Create conda environment and install required packages `conda env create -n allreduce_env -f environment.yml`
3. Activate the environment `conda activate allreduce_env`

## Install PyCharm
https://www.jetbrains.com/pycharm/download/#section=windows

## Modify allreduce implementation
Search for `# Modify gradient allreduce here` and update code there. Replace star-reduce code with ring-allreduce as per:
1. https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2
2. https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da

## Run tests to make sure that weight all-reduce gives correct results
```bash
python test_dataparallel.py
```

## Run neural network training on MNIST datastet
Training is run for the reference and dataparallel models simultaneously
```bash
python train_model.py
```