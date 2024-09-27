import torch.nn as nn
from .Classification_Module import Classification_Module
from .TS_Mixer import Temporal_Mixer
from .Temporal_Encoder import TemporalEncoder
import torch


class MSTCT(nn.Module):
    """
    MS-TCT for action detection
    """

    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes, input_size):
        super(MSTCT, self).__init__()

        self.dropout = nn.Dropout()
        self.projection_layer_clip = None
        if input_size != in_feat_dim:
            self.projection_layer_clip = nn.Linear(
                input_size, in_feat_dim)

        self.TemporalEncoder = TemporalEncoder(
            in_feat_dim=in_feat_dim,
            embed_dims=inter_channels,
            num_head=head,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            num_block=num_block
        )

        self.Temporal_Mixer = Temporal_Mixer(
            inter_channels=inter_channels,
            embedding_dim=final_embedding_dim
        )

        self.Classification_Module = Classification_Module(
            num_classes=num_classes,
            embedding_dim=final_embedding_dim
        )

    def forward(self, inputs_i3d, inputs_clip):

        inputs_i3d = self.dropout(inputs_i3d)
        inputs_clip = self.dropout(inputs_clip)

        if self.projection_layer_clip:
            inputs_clip = inputs_clip.transpose(1, 2)
            inputs_clip = self.projection_layer_clip(inputs_clip)
            inputs_clip = inputs_clip.transpose(1, 2)

        combined_features = torch.cat([inputs_i3d, inputs_clip], dim=2)

        x = self.TemporalEncoder(combined_features)

        concat_feature, concat_feature_hm = self.Temporal_Mixer(x)

        x, x_hm = self.Classification_Module(concat_feature, concat_feature_hm)

        return x, x_hm
