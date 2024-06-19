import argparse
from dgl.nn.pytorch import MetaPath2Vec
from openhgnn.dataset.NodeClassificationDataset import OHGB_NodeClassification

if __name__ == '__main__':
    acm = OHGB_NodeClassification(
        dataset_name="ohgbn-acm", raw_dir="./dataset", logger=None
    )
    hg = acm.g
    meta_paths_dict = acm.meta_paths_dict



# print(parser)
