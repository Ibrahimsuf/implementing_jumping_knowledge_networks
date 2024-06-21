from torch_geometric.data import Data
from typing import Optional
import torch
def split_data_node_classification(graph: Data, train_ratio: float = 0.8, val_ratio: float = 0.1, manual_seed: Optional[int] = None) -> None:
  """
  Create train, val, and test masks for node classification task on a graph. Modifies the graph in place.
  Args:
    graph: torch_geometric.data.Data
    train_ratio: float
    val_ratio: float
  Returns:
    None
  """
  # create 60, 20, 20 train, validation, test splits
  num_nodes = graph.num_nodes
  num_train = int(num_nodes * 0.6)
  num_val = int(num_nodes * 0.2)
  if manual_seed is not None:
    torch.manual_seed(manual_seed)

  perm = torch.randperm(num_nodes)
  train_idx = perm[:num_train]
  val_idx = perm[num_train:num_train + num_val]
  test_idx = perm[num_train + num_val:]

  graph.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
  graph.train_mask[train_idx] = True

  graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
  graph.val_mask[val_idx] = True

  graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
  graph.test_mask[test_idx] = True

def train(model: torch.nn.Module, graph: Data, epochs: Optional[int] = 200, learning_rate: Optional[float] = 0.005) -> None:
  """
  Train a model on a graph. Trains the model in place.
  Args:
    model: torch.nn.Module 
    graph: torch_geometric.data.Data
    epochs: int
    learning_rate: float
  Returns:
    None
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
  for epoch in range(epochs):
    out = model(graph.x, graph.edge_index)
    loss = torch.nn.functional.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
      print(f'Epoch: {epoch}, Loss: {loss.item()}')
def test(model: torch.nn.Module, graph: Data, mask: Optional[torch.Tensor] = None) -> float:
  """
  Test a model on a graph using the test mask.
  Args:
    model: torch.nn.Module
    graph: torch_geometric.data
    mask: torch.Tensor the test mask defaults to graph.test_mask
  Returns:
    acc: float the accuracy of the model on the test mask
  """
  if mask is None:
    mask = graph.test_mask
  model.eval()
  out = model(graph.x, graph.edge_index)
  pred = out.argmax(dim=1)
  correct = (pred[mask] == graph.y[mask]).sum()
  acc = int(correct) / int(mask.sum())
  return acc