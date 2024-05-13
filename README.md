# codes for the paper 《Robust Network Alignment with the Combination of Structure and Attribute Embeddings》.
# dependencies

pytorch 1.10.2，networkx 2.3，python 3，cuda 10.2，dgl 1.1.0+cu102

# file construction

* algorithms
  * GATv2：contains the embedding model: `multi_GATv2.py` and  `embedding.py`, the final run file `run_method.py` 
  * refine: code for topological refinement
* evaluation: contains the evaluation methods for the models
* graph data: contain `Douban` and `PPI` datasets.
* input: to generate the graph data
* utils: contains some files for add noise to graph and generate target graphs for synthetic datas.

# To run method

Go to `MyAlign\algorithms\MyMethod\GATv2\run_method.py`, and choose the data you want to test to run.

