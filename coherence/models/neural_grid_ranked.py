import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

rs = np.random.RandomState(1234)
random.seed(1234)

def perturb_grids(grids_dict, n):
    """Perturb a grid n times."""

    grids = grids_dict['tokens'].clone().detach()
    # grids.shape == [batch_size, n_entities, n_sentences]

    batch_size = grids.shape[0]
    for i in range(batch_size):
        grid = grids[i]
        n_entities, n_sents = grid.shape

        for _ in range(n):
            # Select a random entity and turn it off
            entity_idx = rs.choice(n_entities)
            sent_indices = rs.permutation(n_sents)
            for sent_idx in sent_indices:
                role = grid[entity_idx, sent_idx]
                if role <= 1 or role >= 5:
                    continue
                else:
                    grid[entity_idx, sent_idx] = 5
                    break

    return {'tokens': grids}

def randomize_grids(grids_dict, p=0.2):
    """Randomly move cells around."""
    grids = grids_dict['tokens'].clone().detach()
    # grids.shape == [batch_size, n_entities, n_sentences]

    # Randomly select entities to switch
    seeds = rs.rand(*grids.shape)
    mask = grids.new_tensor((seeds < p).astype(np.uint8)).byte()
    entity_masks = mask & (grids >= 2) & (grids <= 4)

    # Save list of empty locations
    non_masks = (grids == 5)

    # Turn all selected entities off
    for b, entity_mask in enumerate(entity_masks):
        non_mask = non_masks[b]
        non_mask_locations = non_mask.nonzero()
        non_mask_indices = list(range(len(non_mask_locations)))
        random.shuffle(non_mask_indices)

        for i, j in entity_mask.nonzero():
            # Save the entity
            role = grids[b, i, j]

            # Turn entity off
            grids[b, i, j] = 5

            # Randomly select an empty location
            if not non_mask_indices:
                continue
            loc_index = non_mask_indices.pop()
            x, y = non_mask_locations[loc_index]

            # Move this entity to the new location
            grids[b, x, y] = role

    return {'tokens': grids}

def shuffle_grids(grids_dict):
    """Perturb a grid n times."""

    grids = grids_dict['tokens'].clone().detach()
    # grids.shape == [batch_size, n_entities, n_sentences]

    batch_size = grids.shape[0]
    for i in range(batch_size):
        n_sents = grids.shape[-1]
        sent_indices = rs.permutation(n_sents)
        grids[i] = grids[i, :, sent_indices]

    return {'tokens': grids}

def add_to_grid(grids_dict, n):
    """Perturb a grid n times."""

    grids = grids_dict['tokens'].clone().detach()
    # grids.shape == [batch_size, n_entities, n_sentences]

    batch_size, n_entities, n_sents = grids.shape

    n_new = rs.randint(1, n+1)
    for b in range(batch_size):
        for _ in range(n_new):
            done = False
            for _ in range(10):
                entity_idx = rs.choice(n_entities)
                sent_indices = rs.permutation(n_sents)
                for sent_idx in sent_indices:
                    role = grids[b, entity_idx, sent_idx]
                    if role == 5:
                        grids[b, entity_idx, sent_idx] = int(rs.choice([2, 3, 4]))
                        done = True
                        break
                if done:
                    break

    return {'tokens': grids}


@Model.register('ranked_neural_grid')
class RankedNeuralGridModel(Model):
    def __init__(self,
                 vocab: Vocabulary):
        super().__init__(vocab)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion_1 = nn.MarginRankingLoss(margin=0.1, reduction='mean')
        # self.criterion_2 = nn.MarginRankingLoss(margin=1, reduction='mean')
        self.n_samples = 0
        self.json_metrics: Dict[str, list] = defaultdict(list)

        self.n_pos, self.n_noisy, self.n_extra = 0, 0, 0
        self.pos_score, self.noisy_score, self.extra_score = 0, 0, 0

        self.embedder = nn.Embedding(6, 768, padding_idx=0)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(768, 768, 3, padding=1),
            nn.Conv1d(384, 384, 5, padding=2),
            nn.Conv1d(256, 256, 7, padding=3),
        ])
        self.fcs = nn.ModuleList([
            nn.Linear(768, 384),
            nn.Linear(384, 256),
            nn.Linear(256, 128),
        ])
        self.final_fc = nn.Linear(128, 1)

    def forward(self,  # type: ignore
                grids: Dict[str, torch.Tensor],
                embeds: torch.Tensor,
                names: List[List[str]],
                targets: torch.Tensor,
                split: List[str]) -> Dict[str, torch.Tensor]:
        # 0 is padding, 1 is OOV, 2-5 are S/O/X/-
        # grids['tokens'].shape == [batch_size, n_entities, n_sentences]
        # embeds.shape == [batch_size, n_entities, 768]
        # targets.shape == [batch_size]
        # names is a simple Python list of lists

        # First feed to positive examples
        logits_0 = self.feedfoward(grids, embeds)
        batch_size = logits_0.shape[0]

        # Randomly turn off an entity in a sentence
        grids_1 = randomize_grids(grids, p=0.2)
        logits_1 = self.feedfoward(grids_1, embeds)

        # Add up to 5 random entity mentions
        grids_2 = add_to_grid(grids, n=5)
        logits_2 = self.feedfoward(grids_2, embeds)

        # Margin ranking loss. When targets are 1, the first input should be
        # ranked higher.
        probs_0 = torch.sigmoid(logits_0)
        probs_1 = torch.sigmoid(logits_1)
        probs_2 = torch.sigmoid(logits_2)

        targets = logits_0.new_ones(batch_size)
        # loss = self.criterion_1(logits_0, logits_1, targets)
        loss = self.criterion_1(probs_0, probs_1, targets)
        loss += self.criterion_1(probs_0, probs_2, targets.clone().detach())

        # probs_0 = F.log_softmax(logits_0, dim=-1)
        # probs_1 = F.log_softmax(logits_1, dim=-1)

        # We want the original version to have a higher score, so minimize
        # loss = (probs_1[:,1] - probs_0[:,1]).mean()

        # hinge loss, if we don't cut off at 0, the loss will just explode.
        # loss = torch.max(0, 1 - logits_0 + logits_1).mean()

        # if not self.training:
        #     self.json_metrics['probs_0'] += probs_0.tolist()
        #     self.json_metrics['probs_1'] += probs_1.tolist()

        output_dict = {'loss': loss, 'sample_size': torch.tensor(batch_size)}
        self.n_pos += probs_0.shape[0]
        self.n_noisy += probs_1.shape[0]
        self.n_extra += probs_2.shape[0]

        self.pos_score += probs_0.sum().item()
        self.noisy_score += probs_1.sum().item()
        self.extra_score += probs_2.sum().item()

        self.n_samples += batch_size

        if not self.training:
            self.json_metrics['pos'] += probs_0.tolist()
            self.json_metrics['noisy'] += probs_1.tolist()
            self.json_metrics['extra'] += probs_2.tolist()

        return output_dict

    def predict(self, grids, embeds):
        logits = self.feedfoward(grids, embeds)
        # probs = torch.sigmoid(logits)
        return logits

    def feedfoward(self, grids, embeds):
        tokens = grids['tokens']
        X = self.embedder(tokens)
        # X.shape == [batch_size, n_entities, n_sentences, 768]

        X = X.transpose(2, 3)
        batch_size, n_entities, embed_size, n_sents = X.shape
        X = X.view(-1, embed_size, n_sents)
        # X.shape == [batch_size * n_entities, 768, n_sentences]

        embeds = embeds.view(-1, embed_size)
        # embeds.shape == [batch_size * n_entities, 768]

        embeds = embeds.unsqueeze(-1)
        # embeds.shape == [batch_size * n_entities, 768, 1]

        X = X + embeds
        # X.shape == [batch_size * n_entities, 768, n_sentences]

        pad_mask = tokens.eq(0)
        pad_mask = pad_mask.view(-1, n_sents)
        # pad_mask.shape == [batch_size * n_entities, n_sentences]

        pad_mask = pad_mask.unsqueeze(1)
        # pad_mask.shape == [batch_size * n_entities, 1, n_sentences]

        for conv, fc in zip(self.convolutions, self.fcs):
            residual = X

            pad_mask_i = pad_mask.expand(*X.shape)
            X[pad_mask_i] = 0

            X = F.dropout(X, p=0.1, training=self.training)
            X = conv(X)
            X = F.relu(X)
            X = F.dropout(X, p=0.1, training=self.training)
            X = X + residual
            # X.shape == [batch_size * n_entities, 768, n_sentences]

            X = fc(X.transpose(1, 2)).transpose(1, 2)

        X = X.mean(dim=-1)
        # X.shape == [batch_size * n_entities, 768]

        X = X.view(batch_size, n_entities, -1)
        # X.shape == [batch_size, n_entities, 768]

        X = X.mean(dim=1)
        # X.shape == [batch_size, 768]

        logits = self.final_fc(X)
        # logits.shape == [batch_size, 2]

        return logits

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics['n_samples'] = self.n_samples

        if self.n_pos > 0:
            all_metrics['pos'] = self.pos_score / self.n_pos
        if self.n_noisy > 0:
            all_metrics['noisy'] = self.noisy_score / self.n_noisy
        if self.n_extra > 0:
            all_metrics['extra'] = self.extra_score / self.n_extra

        if reset:
            self.n_pos, self.n_noisy, self.n_extra = 0, 0, 0
            self.pos_score, self.noisy_score, self.extra_score = 0, 0, 0
            self.n_samples = 0

        return all_metrics

    def get_json_metrics(self, reset: bool = False) -> Dict[str, Any]:
        all_metrics: Dict[str, Any] = {}
        for metric, histogram in self.json_metrics.items():
            all_metrics[metric] = histogram

        if reset:
            self.json_metrics = defaultdict(list)

        return all_metrics

    @classmethod
    def from_params(cls: Type['RankedNeuralGridModel'],
                    vocab: Vocabulary,
                    params: Params) -> 'RankedNeuralGridModel':
        logger.info(f"instantiating class {cls} from params {getattr(params, 'params', params)} "
                    f"and vocab {vocab}")

        return cls(vocab=vocab)
