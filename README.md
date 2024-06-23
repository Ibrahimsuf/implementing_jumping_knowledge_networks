# Implementing Jumping Knowledge Networks

Attempts to Replicate the results from  [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536) which introduces a new graph neural network architecture where the final layer aggegrates information from all previous hidden layers.

## Installation Instruction

1. `git clone https://github.com/Ibrahimsuf/implementing_jumping_knowledge_networks.git`
2. `cd implementing_jumping_knowledge_networks`
3. create conda environment `conda create --name jumping_knowledge_networks`
4. activate the environment `conda activate jumping_knowledge_networks`
5. `conda install pip`
6. `pip install .`
7. Run the code in the Experiments Juypter Notebook.

## Future Work

At some point I would like to finish testing on the other datasets used in the paper as well as testing other aggregation methods other than concatination.
