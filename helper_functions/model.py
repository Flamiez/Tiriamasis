import torch.nn as nn
import torchvision.models as models

class FeatureExtractModel(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(FeatureExtractModel, self).__init__()
        googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        
        self.base = nn.Sequential(
            googlenet.conv1,
            googlenet.maxpool1,
            googlenet.conv2,
            googlenet.conv3,
            googlenet.maxpool2,
            googlenet.inception3a,
            googlenet.inception3b,
            googlenet.maxpool3,
            googlenet.inception4a,
            googlenet.inception4b,
            googlenet.inception4c,
            googlenet.inception4d,
            googlenet.inception4e,
            googlenet.maxpool4,
            googlenet.inception5a,
            googlenet.inception5b,
            googlenet.avgpool
        )
        
        self._add_bn_layers()
        self.embedding = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()
    
    def _add_bn_layers(self):
        for name, module in self.base.named_children():
            if 'inception' in name:
                module.add_module('bn', nn.BatchNorm2d(module.output_channels))
    
    def _initialize_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        norm_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return norm_embedding 