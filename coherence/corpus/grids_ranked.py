import logging
import pickle
import random
from typing import Dict, Iterable, List

import joblib
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import (ArrayField, LabelField, ListField,
                                  MetadataField, TextField)
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers.token import Token

from .base import Corpus, FlexibleDatasetReader

logger = logging.getLogger(__name__)

idx2role = ['-', 'S', 'O', 'X']


@Corpus.register('grids_ranked')
class RankedEntityGridCorpus(Corpus):
    def __init__(self,
                 data_dir: str,
                 lazy: bool,
                 indexer: Dict[str, TokenIndexer],
                 transpose: bool):
        super().__init__()

        grid_path = f'{data_dir}/entity_grids_with_embeds.pkl'
        logger.info(f'Loading entity grids from {grid_path}')
        with open(grid_path, 'rb') as f:
            grids, entities, embeds = pickle.load(f)

        reader = EntityGridReader(lazy, indexer, transpose)

        logger.info(f'Preparing training data')
        self.train = reader.read(grids['train'], entities['train'],
                                 embeds['train'])

        logger.info(f'Preparing validation data')
        self.valid = reader.read(grids['valid'], entities['valid'],
                                 embeds['valid'])

        logger.info(f'Preparing test data')
        self.test = reader.read(grids['test'], entities['test'],
                                embeds['test'])


class EntityGridReader(FlexibleDatasetReader):
    def __init__(self, lazy, indexer: Dict[str, TokenIndexer], transpose: bool):
        super().__init__(lazy=lazy)
        self.indexer = indexer
        self.transpose = transpose

    def get_positive_instance(self, grid, entities, embeds, rs) -> Instance:
        if self.transpose:
            grid = grid.transpose()
        n_sents = rs.randint(1, 50)
        grid = grid[:, :n_sents]
        grid_field = [TextField(self.get_transition_tokens(transitions), self.indexer)
                      for transitions in grid]

        instance = Instance({
            'names': MetadataField(entities),
            'embeds': ArrayField(embeds),
            'grids': ListField(grid_field),
            'targets': LabelField(1, skip_indexing=True),
            'split': MetadataField('positive'),
        })

        return instance


    def get_transition_tokens(self, transitions):
        return [Token(idx2role[i]) for i in transitions]

    def get_tokens(self, transitions):
        return [Token(s) for s in transitions]

    def _read(self, grids, entities, embeds) -> Iterable[Instance]:
        rs = np.random.RandomState(1234)
        random.seed(1234)

        # Shuffle datasets
        indices = sorted(entities.keys())
        rs.shuffle(indices)

        for i in indices:
            if i not in embeds:
                continue
            yield self.get_positive_instance(grids[i], entities[i], embeds[i], rs)
