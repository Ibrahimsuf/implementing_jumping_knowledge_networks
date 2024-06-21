import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from typing import List, Optional
import torch.nn.functional as F
import torch.nn.functional as F
class GCN_JK_Concat(torch.nn.Module):
  def __init__(self, starting_features: int, hidden_channels: List[int], output_features: int, agg_method: str = 'concat'):
    super().__init__()
    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(starting_features, hidden_channels[0]))
    self.agg_method = agg_method
    for input_channels, output_channels in zip(hidden_channels[:-1], hidden_channels[1:]):
      self.convs.append(GCNConv(input_channels, output_channels))

    
    if self.agg_method == 'concat':
      total_channels = sum(hidden_channels) + starting_features
      self.lin = Linear(total_channels, output_features)
    elif self.agg_method == 'max pool':
      if all([channels_num == hidden_channels[0] for channels_num in hidden_channels]):
        self.lin = Linear(hidden_channels[0], output_features)
      else:
        raise ValueError('Max pooling requires all hidden channels to be equal')
  def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    layer_outputs = [x] if self.agg_method == 'concat' else []
    h_i = x
    for conv in self.convs:
      h_i = conv(h_i, edge_index)
      h_i = F.dropout(h_i, p=0.5, training=self.training)
      h_i = h_i.relu()
      layer_outputs.append(h_i)
    
    if self.agg_method == 'concat':
      agg = torch.cat(layer_outputs, dim=1)
    elif self.agg_method == 'max pool':
      agg = torch.stack(layer_outputs).max(dim=0)[0]

    return self.lin(agg)


