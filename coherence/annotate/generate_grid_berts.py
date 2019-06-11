"""CoreNLP Annotation

Usage:
    generate_grids.py [--repo FILE]
    generate_grids.py (-h | --help | --version)

Options:
    -h --help      Show this screen.
    --version      Show version.
    --repo FILE    The path of the output repo
                   [default: data/film-repo-eventized.plk].
"""


import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from bert_serving.client import BertClient
from docopt import docopt
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    bert_client = BertClient('localhost')
    data_dir = f'./data/writing-prompts'
    grid_path = f'{data_dir}/entity_grids.pkl'
    with open(grid_path, 'rb') as f:
        grids, entities = pickle.load(f)

    entity_embeds = defaultdict(dict)
    for split in ['train', 'valid', 'test']:
        logger.info(f'Getting entity embedding for {split}')
        for i, names in tqdm(entities[split].items()):
            if names:
                entity_embeds[split][i] = bert_client.encode(names)

    out_path = f'{data_dir}/entity_grids_with_embeds.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump((grids, entities, entity_embeds), f)


if __name__ == '__main__':
    main()
