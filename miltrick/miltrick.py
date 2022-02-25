from numpy import iterable
import torch
import abc
from typing import Iterable, Tuple
from torch.utils.data.dataset import Dataset



#def split_into_batches(lst: Iterable, n):
#    return [lst[i:i + n] for i in range(0, len(lst), n)]


def split_into_batches(iterable, n=1):
    l = len(iterable)
    return [iterable[ndx:min(ndx + n, l)] for ndx in range(0, l, n)]
    

class MILModel(torch.nn.Module):
    '''
        The MILModel consists of two parts:
            1. A feature extractor model that, given 
                a patch, or batch of patches, outputs a 
                feature vector.
            2. An attention model that that outputs a
                prediction given a set of feature vectors.
    '''

    def __init__(self, 
        feature_extractor:torch.nn.Module, 
        attention_model: torch.nn.Module,
        loss_fun
        ):

        super().__init__()
        self.feature_extractor = feature_extractor
        self.is_feature_extractor_frozen = False

        self.attention_model = attention_model
        self.is_attention_model_frozen = False
        self.loss_fun = loss_fun

                

    def _freeze_feature_extractor(self):
        if not self.is_feature_extractor_frozen:
            for params in self.feature_extractor.parameters():
                params.requires_grad = False
            
            for layer in self.feature_extractor.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                    layer.track_running_stats = False

            self.is_feature_extractor_frozen = True

    def _unfreeze_feature_extractor(self):
        if self.is_feature_extractor_frozen:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            for layer in self.feature_extractor.modules():
                if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                    layer.track_running_stats = True
            self.is_feature_extractor_frozen = False



    def _freeze_attention_model(self):
        if not self.is_attention_model_frozen:
            for param in self.attention_model.parameters():
                param.requires_grad = False
            self.is_attention_model_frozen = True

    def _unfreeze_attention_model(self):
        if self.is_attention_model_frozen:
            for param in self.attention_model.parameters():
                param.requires_grad = True
            self.is_attention_model_frozen = False            

    def _forward_pass_train(self, bag, target, scale_loss):
        x = self.feature_extractor(bag)
        pred, attention = self.attention_model(x)
        l = self.loss_fun(pred, target) * scale_loss
        l.backward()
        return pred, attention, l.item()

    def _incremental_forward_pass_train(self, bag, target, max_instances_per_forward_pass,scale_loss):
        # REFACTOR THIS LINE AXEL
        batchind2featureind_lut = [
            torch.arange(ndx,min(ndx+max_instances_per_forward_pass,len(bag))) 
            for ndx in range(0, len(bag), max_instances_per_forward_pass)
        ]

        bag = split_into_batches(bag, max_instances_per_forward_pass)
            
        # Freeze gradients
        self._freeze_feature_extractor()
        self._freeze_attention_model()
        feature_vectors = self.__compute_feature_vectors(bag)

        # self.feature_extractor.layer2[0].conv1.weight
        # self.attention_model.attention[0].weight
        # Unfreeze attention model
        self._unfreeze_attention_model()
        pred, attention = self.attention_model(feature_vectors)
        l = self.loss_fun(pred, target) * scale_loss
        l.backward()

        # Set the gradient for the feature extractor
        self._freeze_attention_model()
        self._unfreeze_feature_extractor()
        identity_matrix = torch.eye(feature_vectors.shape[0], dtype=torch.bool)

        for batch_of_patches, patch_idx in (zip(bag, batchind2featureind_lut)):
            fv = self.feature_extractor(batch_of_patches)
            other_patches_idx = ~torch.max(identity_matrix[patch_idx], dim=0)[0]
            other_fv = feature_vectors[other_patches_idx]
            pred, _ = self.attention_model(torch.cat( (fv, other_fv)))
            l = self.loss_fun(pred, target) * scale_loss
            l.backward()

        return pred, attention, l.item()


    def _forward_pass_test(self, bag, target, scale_loss):
        x = self.feature_extractor(bag)
        pred, attention = self.attention_model(x)
        if target is None:
            return pred, attention, None
        else:
            l = self.loss_fun(pred, target) * scale_loss
            return pred, attention, l.item()

    def _incremental_forward_pass_test(self, bag, target, max_instances_per_forward_pass, scale_loss):
        bag = split_into_batches(bag, max_instances_per_forward_pass)
        feature_vectors = self.__compute_feature_vectors(bag)
        pred, attention = self.attention_model(feature_vectors)
        if target is None:
            return pred, attention, None
        else:
            l = self.loss_fun(pred, target) * scale_loss
            return pred, attention, l.item()

    def forward_train(self, bag, target, max_instances_per_forward_pass:int=None, scale_loss=1.0):
        assert self.training, "Model must be put in train mode. Use model.train()."
        if max_instances_per_forward_pass:
            return self._incremental_forward_pass_train(bag, target, max_instances_per_forward_pass, scale_loss)
        else:
            return self._forward_pass_train(bag, target, scale_loss)


    def forward_test(self, bag, target=None, max_instances_per_forward_pass:int=None, scale_loss=1.0):
        assert not self.training, "Model must be put in test mode. Use model.eval()."
        if max_instances_per_forward_pass is None:
            return self._forward_pass_test(bag, target, scale_loss)
        else:
            return self._incremental_forward_pass_test(bag, target, max_instances_per_forward_pass, scale_loss)


    def __compute_feature_vectors(self, bag):
        with torch.no_grad():
            fv = []
            for batch_of_patches in bag:
                fv.append(self.feature_extractor(batch_of_patches))
            return torch.cat(fv)

