# Automatic Detection of Sacroiliitis in MRI using a Stacked Ensemble of Convolutional Neural Networks.

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
<li><a href="https://pytorch.org/docs/stable/index.html">PyTorch docs</a></li>
<li><a href="https://pytorch.org/vision/stable/index.html">Torchvision</a> </li>
<li><a href="https://pytorch-lightning.readthedocs.io/en/1.3.8/">Pytorch Lightning Docs</a> </li>
</ol>

Models Used (and modified):
<ul>
<li><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html?highlight=resnet18#torchvision.models.resnet18">PyTorch ResNet18</a> </li>
<li><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html#torchvision.models.densenet121">PyTorch DenseNet121</a> </li>
<li><a href="SIJEnsemble.py">Ensemble model</a> </li>
</ul>
