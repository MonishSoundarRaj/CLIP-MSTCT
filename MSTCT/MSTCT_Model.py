import torch.nn as nn
from .Classification_Module import Classification_Module
from .TS_Mixer import Temporal_Mixer
from .Temporal_Encoder import TemporalEncoder
import numpy as np


class MSTCT(nn.Module):
    """
    MS-TCT for action detection
    """
    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes, input_size):
        super(MSTCT, self).__init__()

        self.dropout=nn.Dropout()

        self.projection_layer = None
        if input_size != in_feat_dim:
            self.projection_layer = nn.Linear(input_size, in_feat_dim)

        self.TemporalEncoder=TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                 num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,num_block=num_block)

        self.Temporal_Mixer=Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        #print(inputs.shape)

        if self.projection_layer:
            # inputs.shape is (batch_size, 768, num_frames)
            inputs = inputs.transpose(1, 2)  # Shape: (batch_size, num_frames, 768)

            inputs = self.projection_layer(inputs)  # Shape: (batch_size, num_frames, 1024)

            inputs = inputs.transpose(1, 2)  # Shape: (batch_size, 1024, num_frames)
            

        # Temporal Encoder Module
        x = self.TemporalEncoder(inputs)

        # Temporal Scale Mixer Module
        concat_feature, concat_feature_hm = self.Temporal_Mixer(x)

        # Classification Module
        x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)

        return x, x_hm # B, T, C





