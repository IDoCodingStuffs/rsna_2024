import torch
import timm_3d 
import torch.nn as nn

from config import Config

class GradingModel(nn.Module):
    def __init__(self,
                 backbone='resnet3d_50',  # Example 3D CNN backbone
                 in_chans=3,
                 out_classes=5,
                 cutpoint_margin=0):
        super(GradingModel, self).__init__()

        # Create the 3D CNN backbone using timm_3d
        self.backbone = timm_3d.create_model(
            Config.backbone,
            pretrained=True,
            in_chans=3,
            num_classes=0  # We set this to 0 to exclude the final classification layer
        )
        
        # Extract the number of features from the backbone's last layer
        head_in_dim = self.backbone.num_features

        # Define the final classification head
        self.num_classes = out_classes
        self.logits = nn.Linear(head_in_dim, self.num_classes)
    
    def extract_features(self, x):
        # Forward pass through the backbone to extract features
        return self.backbone(x)
    
    def forward(self, x):
        # Pass the input through the backbone to get features
        fts = self.extract_features(x)
        
        # Apply the classification head to the features
        return self.logits(fts)
