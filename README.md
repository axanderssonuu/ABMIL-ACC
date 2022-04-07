# ABMIL with accumulating gradients

## What?
Being able to learn on weakly labeled data, and provide interpretability, are two of the main reasons why attention-based deep multiple instance learning (ABMIL) methods have become particularly popular for classification of histopathological images. Such image data usually come in the form of gigapixel-sized whole-slide-images (WSI) that are cropped into smaller patches (instances). However, the sheer size of the data makes training of ABMIL models challenging.

Here we provide a simple example on how to train ABMIL-models without constraints on the GPU memory. Memory limitations are circumvented by strategically accumulating gradients over individual instances, thus avoiding the need for placing entire bags on the GPU.

## Usage:

To use the gradient accumulation trick the user must first create a `MILModel` class:
```python
from miltrick.miltrick import MILModel
mil_model = MILModel(feature_encoder, attention_model, loss_fun)
```
The `MILModel` class takes three input arguments to its constructor:
* `feature_encoder`: A feature encoder network implemented in PyTorch. This is the bottle-neck of the architecture and is often chosen as a type of ResNET. This network is used to transform individual instances into lower dimensional feature representations. See `qmnist_models.py` for example  of implementation.
* `attention_model`:  An attention network that given a set of feature representations, stored as an `(nFeatureDim,nInstances)` tensor, pools the feature representations and predicts a bag level. See `qmnist_models.py` for example  of implementation.
* `loss_fun`: Loss function used for training.

The `MILModel` can then be trained end-to-end by calling the `forward_train` method:

```python
model.forward_train(
    bag, 
    bag_label, 
    max_instances_per_forward_pass=10, 
)
```
The optional parameter `max_instances_per_forward_pass` sets the maximum number instances that will processed simultanously during training. If set to `Ǹone` the entire bag will be processed simultanously.