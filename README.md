# Automatic Detection of Sacroiliitis in MRI using a Stacked Ensemble of Convolutional Neural Networks.

## About
This project is the code for my dissertation project for my Master's of Science (MS) in Computer Science with Artificial Intelligence at the University of Wolverhampton.
The original code was written in Jupyter and later converted to python files.

## Running on your machine
If you want to run this program on your machine, you'll need to follow these steps:
1. Create your own .env file with the environment variables listed in main.py
2. run the program initially from your command line:
```
python main.py train save
```
3. To show the results of the final ensemble classifier run the following from your command line:
```
python main.py results
```

Step 3 will print out in your terminal the accuracy, precision, recall and a confusion matrix.

## System Information
### Device
In main.py, I have set the device to 'mps' because I am using an M1 mac.
If you are using a cuda compatible device, you will need to change all mps references in main.py to use cuda instead.

### System Requirements
Aside from the libraries imported in each of the files, you will need to have python 3.10 or higher to run this project.
this project was developed using a conda environment, so you might want to download Anaconda and create an environment in which you can install the relevant packages.

### Data
The dataset for this project is not publicly available. Due to the license I cannot share the data publicly.
The format of the data in the Dataset comes in .dcm (dicom), .png and .psd formats. Of these formats, dicom was used.
During the initial preprocessing of the data (not included in this repo) the dicom images were converted to NumPy arrays.

## Models and Libraries
The libraries used to build, train and validate the models was PyTorch, TorchVision and PyTorch Lightning.
The user documentation can be found at:
<ol>
<li><a href="https://pytorch.org/docs/stable/index.html" target="_blank">PyTorch docs</a></li>
<li><a href="https://pytorch.org/vision/stable/index.html" target="_blank">Torchvision</a> </li>
<li><a href="https://pytorch-lightning.readthedocs.io/en/1.3.8/" target="_blank">Pytorch Lightning Docs</a> </li>
</ol>

Models Used (and modified):
<ul>
<li><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html?highlight=resnet18#torchvision.models.resnet18" target="_blank">PyTorch ResNet18</a> </li>
<li><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html#torchvision.models.densenet121" target="_blank">PyTorch DenseNet121</a> </li>
<li><a href="SIJEnsemble.py">Ensemble model</a> </li>
</ul>
