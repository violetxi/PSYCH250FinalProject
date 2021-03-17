# PSYCH250FinalProject
The goal of this project is to investigate wheter a DCNN has category-specific representations. To do this, we trained a CNN to learn **face**, **body**, **places** and **characters**, and add lesion to the network after it has learned these categories, and check their overall and class-specific classification abilities. 

## Model & Experiment Implementations
We used PyTorch's AlexNet pre-trained on ImageNet, and then we replaced the readout layer with a fully-connected layer with 4 output units. `model.py`
We implemented our data pipeline as standard PyTorch Dataset classes and load it with Dataloader, and we used standard ImageNet preprocessing by randomly resizing images and cropping them at 224x224, then randomly flipping some of them horizontally finally normalized images with ImageNet mean and std. `dataset.py`
Training is implemented in `trainer.py`. Training is done for 10 epochs and checkpoints are saved at each epoch.
Evaluation and lesioning are implemented in `run_exp.py`. This module first check initial oever performance and performance by each category for each checkpoint. Then it randomly selects 10% of the filters in each of the **five** convolutional layers to be lesioned 10 times. We check both overall and category-specific performance for each time, and save the results.

*For results visualization, please check the notebook in writeup/ folder*.

## Command for running experiments
Training: `python trainer.py --num_epochs 20 --batch_size 128 --init_lr 1e-4 --save_freq 4 --result_folder checkpoints/ --meta_path data/processed/meta.pkl`

Experiments: `python run_exp.py --ckpt_folder checkpoints/ --meta_path data/processed/meta.pkl --result_folder results/`
