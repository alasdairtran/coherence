---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
import pickle
import copy
import random
import torch

from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import (ArrayField, LabelField, ListField,
                                  MetadataField, TextField)
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util

import coherence
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict

random.seed(123)
rs = np.random.RandomState(123)
```

```python
archive = load_archive('./experiments/entity_grid_ranking/model.tar.gz', 0, overrides={})
config = archive.config
prepare_environment(config)
model = archive.model
model.eval()

data_dir = './data/writing-prompts'
grid_path = f'{data_dir}/entity_grids_with_embeds.pkl'
with open(grid_path, 'rb') as f:
    grids, entities, embeds = pickle.load(f)
```

```python
def shuffle_grid(grid):
    new_grid = copy.deepcopy(grid)
    for i in range(len(new_grid)):
        random.shuffle(new_grid[i])
        
    return new_grid
```

```python
d = ['-', 'S', 'O', 'X']
def get_score(grid, embed):
    grid = [[d[col] for col in row] for row in grid.transpose()]
    
    indexer = {'tokens': TokenIndexer.from_params(Params({
                'type': 'single_id'
    }))}
    grid_field = [TextField([Token(s) for s in transitions], indexer)
                                 for transitions in grid]
    instances = [Instance({
        'embeds': ArrayField(embed),
        'grids': ListField(grid_field),
    })]
    
    with torch.no_grad():
        batch = Batch(instances)
        batch.index_instances(model.vocab)
        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tensor_dict = nn_util.move_to_device(tensor_dict, 0)
        probs = model.predict(**tensor_dict)
    
    return torch.sigmoid(probs).item()
```

```python
def randomize_grids(grid, p=0.2):
    """Randomly move cells around."""
    grid = np.array(grid).copy().transpose()
    # grids.shape == [batch_size, n_entities, n_sentences]

    # Randomly select entities to switch
    seeds = rs.rand(*grid.shape)
    mask = (seeds < p)
    entity_mask = mask & (grid >= 1) & (grid <= 3)

    # Save list of empty locations
    non_mask = (grid == 0)

    # Turn all selected entities off
    non_mask_locations = list(zip(*non_mask.nonzero())) 
    non_mask_indices = list(range(len(non_mask_locations)))
    random.shuffle(non_mask_indices)

    for i, j in list(zip(*entity_mask.nonzero())):
        # Save the entity
        role = grid[i, j]

        # Turn entity off
        grid[i, j] = 0

        # Randomly select an empty location
        if not non_mask_indices:
            continue
        loc_index = non_mask_indices.pop()
        x, y = non_mask_locations[loc_index]

        # Move this entity to the new location
        grid[x, y] = role

    return grid.transpose()
```

```python
def perturb_grid(grid, n):
    """Perturb a grid n times."""

    grid = np.array(grid).copy().transpose()
    # grids.shape == [batch_size, n_entities, n_sentences]

    n_entities, n_sents = grid.shape

    for _ in range(n):
        # Select a random entity and turn it off
        entity_idx = rs.choice(n_entities)
        sent_indices = rs.permutation(n_sents)
        for sent_idx in sent_indices:
            role = grid[entity_idx, sent_idx]
            if role == 0:
                continue
            else:
                grid[entity_idx, sent_idx] = 0
                break

    return grid.transpose()
```

```python
def add_to_grid(grid, n):
    """Perturb a grid n times."""

    grid = np.array(grid).copy().transpose()
    # grids.shape == [batch_size, n_entities, n_sentences]

    n_entities, n_sents = grid.shape

    for _ in range(n):
        # Select a random entity and turn it off
        entity_idx = rs.choice(n_entities)
        sent_indices = rs.permutation(n_sents)
        for sent_idx in sent_indices:
            role = grid[entity_idx, sent_idx]
            if role == 0:
                grid[entity_idx, sent_idx] = rs.choice([1,2,3])
                break

    return grid.transpose()
```

```python
def keep_first_sents(grid, embed, n=10):
    """Keep first n sentences."""
    
    # Keep first n sentences
    grid = grid.copy()[:n]
    
    # Mask empty entities
    ent_mask = grid.sum(axis=0) > 0
    
    return grid[:, ent_mask], embed[ent_mask]
```

```python
count = 0
scores_shortened = []
scores_real = []
scores_ent_1 = []
scores_ent_5 = []
scores_swap = []
scores_ent_m1 = []
scores_ent_m2 = []
scores_noisy_5 = []
scores_noisy_10 = []
scores_noisy_20 = []
scores_noisy_30 = []
scores_shuffled = []

for i, grid in tqdm(grids['valid'].items()):
    entity_list = entities['valid'][i]
    if i not in embeds['valid'] or grid.shape[1] < 2 or grid.shape[0] < 2:
        continue
        
    embed = embeds['valid'][i]
    
    # Real score
    score = get_score(grid, embed)
    scores_real.append(score)
    
    # Keep first 10 sentences
    new_grid, new_embed = keep_first_sents(grid, embed, n=10)
    if len(new_embed) == 0:
        continue
    score = get_score(new_grid, new_embed)
    scores_shortened.append(score)
    
    # Add a random entity
    new_grid = add_to_grid(grid, 1)
    score = get_score(new_grid, embed)
    scores_ent_1.append(score)
    
    # Add a random entity
    new_grid = add_to_grid(grid, 5)
    score = get_score(new_grid, embed)
    scores_ent_5.append(score)
    
    # Swap 2 sentences
    new_grid = grid.copy()
    x, y = rs.choice(len(new_grid), 2, replace=False)
    new_grid[[x, y]] = new_grid[[y, x]]
    score = get_score(new_grid, embed)
    scores_swap.append(score)
    
    # Turn off one random entry
    new_grid = perturb_grid(grid, 1)
    score = get_score(new_grid, embed)
    scores_ent_m1.append(score)
    
    # Turn off two random entries
    new_grid = perturb_grid(grid, 2)
    score = get_score(new_grid, embed)
    scores_ent_m2.append(score)

    # Swap entities with 5% prob
    new_grid = randomize_grids(grid, p=0.05)
    score = get_score(new_grid, embed)
    scores_noisy_5.append(score)
    
    # Swap entities with 10% prob
    new_grid = randomize_grids(grid, p=0.1)
    score = get_score(new_grid, embed)
    scores_noisy_10.append(score)
    
    # Swap entities with 20% prob
    new_grid = randomize_grids(grid, p=0.2)
    score = get_score(new_grid, embed)
    scores_noisy_20.append(score)
    
    # Swap entities with 30% prob
    new_grid = randomize_grids(grid, p=0.3)
    score = get_score(new_grid, embed)
    scores_noisy_30.append(score)

    # Randomly permute all sentences
    new_grid = rs.permutation(grid)
    score = get_score(new_grid, embed)
    scores_shuffled.append(score)
    
score_dict = {
    'real': scores_real,
    'ent_1': scores_ent_1,
    'ent_5': scores_ent_5,
    'swap': scores_swap,
    'ent_m1': scores_ent_m1,
    'ent_m2': scores_ent_m2,
    'noisy_5': scores_noisy_5,
    'noisy_10': scores_noisy_10,
    'noisy_20': scores_noisy_20,
    'noisy_30': scores_noisy_30,
    'shuffled': scores_shuffled,
}
with open('ranking_loss_scores.json', 'w') as f:
    json.dump(score_dict, f)
```

```python
rank_count = defaultdict(int)
for i in range(len(scores_real)):
    if scores_real[i] > scores_shortened[i]:
        rank_count['shortened'] += 1
    if scores_real[i] > scores_ent_1[i]:
        rank_count['ent_1'] += 1
    if scores_real[i] > scores_ent_5[i]:
        rank_count['ent_5'] += 1
    if scores_real[i] > scores_swap[i]:
        rank_count['swap'] += 1
    if scores_real[i] > scores_ent_m1[i]:
        rank_count['ent_m1'] += 1
    if scores_real[i] > scores_ent_m2[i]:
        rank_count['ent_m2'] += 1
    if scores_real[i] > scores_noisy_5[i]:
        rank_count['noisy_5'] += 1
    if scores_real[i] > scores_noisy_10[i]:
        rank_count['noisy_10'] += 1
    if scores_real[i] > scores_noisy_20[i]:
        rank_count['noisy_20'] += 1
    if scores_real[i] > scores_noisy_30[i]:
        rank_count['noisy_30'] += 1
    if scores_real[i] > scores_shuffled[i]:
        rank_count['shuffled'] += 1
        
for k, v in rank_count.items():
    rank_count[k] = v / len(scores_real)
```

```python
rank_count
```

```python
raw_scores = {k: np.mean(v) for k, v in score_dict.items()}
raw_scores
```
