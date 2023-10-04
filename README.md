# Automatic Detection of Sacroiliitis in MRI using a Stacked Ensemble of Convolutional Neural Networks.

## About
This project is the code for my dissertation project for my Master's of Science (MS) in Computer Science with Artificial Intelligence at the University of Wolverhampton.
The original code was written in Jupyter and later converted to python files.

## Setup Information
### Device
in main.py, I have set the device to 'mps' because I am using an M1 mac.
If you are using a cuda compatible device, you will need to change line 22 in main.py to use cuda instead of mps.

### Data
The dataset for this project is not publicly available. Due to the license I cannot share the data publicly.
The format of the data in the Dataset comes in .dcm (dicom), .png and .psd formats. Of these formats, dicom was used.
During the preprocessing of the data (not included in this repo) the dicom images were converted to numpy arrays.

### Models and Libraries
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
