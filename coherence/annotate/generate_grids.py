"""CoreNLP Annotation

Usage:
    generate_grids.py
    generate_grids.py (-h | --help | --version)

Options:
    -h --help      Show this screen.
    --version      Show version.
"""


import json
import logging
import os
import pickle
from collections import defaultdict
from glob import glob

import numpy as np
import StanfordDependencies
from docopt import docopt
from joblib import Parallel, delayed
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

logger = logging.getLogger(__name__)

class_paths = os.environ['CLASSPATH'].split(':')
corenlp_path = list(filter(lambda p: p.endswith(
    'stanford-corenlp-3.9.2.jar'), class_paths))[0]
sd = StanfordDependencies.get_instance(
    jar_filename=corenlp_path, backend='jpype')

noun_tags = ['NNP', 'NP', 'NNS', 'NN', 'N', 'NE']
subject_tags = ['csubj', 'csubjpass', 'subj', 'nsubj', 'nsubjpass']
object_tags = ['pobj', 'dobj', 'iobj']

idx2role = ['-', 'X', 'O', 'S']
role2idx = {role: idx for idx, role in enumerate(idx2role)}


def get_deps(sentences):
    entities = defaultdict(lambda: defaultdict(dict))
    parse_trees = []
    for sentence in sentences:
        parse_trees.append(sentence['parse'])

    doc = sd.convert_trees(parse_trees, add_lemmas=True)
    return doc


def get_role(tag):
    if tag in subject_tags:
        return 'S'
    elif tag in object_tags:
        return 'O'
    else:
        return 'X'


def tag_entity(tokens):
    roles = []
    for token in tokens:
        tag = token.deprel
        roles.append(get_role(tag))

    if 'S' in roles:
        return 'S'
    elif 'O' in roles:
        return 'O'
    else:
        return 'X'


def is_new_entity(corefs, sent_idx, start, end):
    for entity, e_corefs in corefs.items():
        for coref in e_corefs:
            if sent_idx != coref['sentNum'] - 1:
                continue
            elif start < coref['startIndex'] - 1:
                continue
            elif end > coref['endIndex'] - 1:
                continue
            else:
                return False
    return True


def is_pronoun(sentences, sent_idx, start, end):
    tokens = sentences[sent_idx]['tokens'][start:end]
    for token in tokens:
        if token['pos'] not in ['PRP', 'PRP$']:  # pronoun or possessive pronoun
            return False
    return True


def get_isolated_entities(corefs, sentences, doc):
    entities = defaultdict(list)
    for sent_idx, sent in enumerate(sentences):
        for mention in sent['entitymentions']:
            # start and end here are already zero-index. end is exclusive
            start = mention['tokenBegin']
            end = mention['tokenEnd']
            text = mention['text']
            is_new = is_new_entity(corefs, sent_idx, start, end)
            if is_new and not is_pronoun(sentences, sent_idx, start, end):
                entity_tokens = doc[sent_idx][start:end]
                role = tag_entity(entity_tokens)
                role_idx = role2idx[role]
                entities[text].append((sent_idx, role_idx))

    return entities


def extract_grid(corefs, sentences, doc):
    isolated_entities = get_isolated_entities(corefs, sentences, doc)
    idx2entity = sorted(corefs)
    entity2idx = {entity: idx for idx, entity in enumerate(idx2entity)}
    n_sentences = len(sentences)

    entity_names = []
    for entity_id in idx2entity:
        entity_names.append(find_rep(corefs, entity_id))

    n_entities = len(entity_names)
    for name in isolated_entities:
        if name not in entity_names:
            n_entities += 1

    grid = np.zeros((n_sentences, n_entities), int)

    for entity, e_corefs in corefs.items():
        for coref in e_corefs:
            #         print(coref)
            sent_idx = coref['sentNum'] - 1
            entity_idx = entity2idx[entity]
            # the startIndex and endIndex both start at 1, so we need to first
            # convert them into zero-index. Also the endIndex is exclusive
            start = coref['startIndex'] - 1
            end = coref['endIndex'] - 1

            entity_tokens = doc[sent_idx][start:end]
            role = tag_entity(entity_tokens)
            role_idx = role2idx[role]

            grid[sent_idx, entity_idx] = max(
                role_idx, grid[sent_idx, entity_idx])

    # Detect existing entities in other mentions
    for name, positions in isolated_entities.items():
        if name in entity_names:
            entity_idx = entity_names.index(name)
            for sent_idx, role_idx in positions:
                grid[sent_idx, entity_idx] = max(
                    role_idx, grid[sent_idx, entity_idx])
        else:  # new entity
            entity_names.append(name)
            entity_idx = len(entity_names) - 1
            for sent_idx, role_idx in positions:
                grid[sent_idx, entity_idx] = max(
                    role_idx, grid[sent_idx, entity_idx])

    # All entities must be mentioned somewhere
    assert np.count_nonzero(grid.sum(axis=0)) == grid.shape[1]

    return grid, entity_names


def extract_all_grids():
    grids = defaultdict(dict)
    entities = defaultdict(dict)

    paths = glob('data/writing-prompts/annotations/*.json')
    for i, path in tqdm(enumerate(paths)):
        try:
            filename = path.split('/')[-1].split('.')[0]
            kind, story_id = filename.split('-')
            with open(path) as f:
                annotation = json.load(f)
            sentences = annotation['sentences']
            corefs = annotation['corefs']
            doc = get_deps(sentences)
            grid, entity_names = extract_grid(corefs, sentences, doc)
            grids[kind][story_id] = grid
            entities[kind][story_id] = entity_names
        except KeyError as e:
            logger.error(f'Key Error processing {kind}-{story_id}: {e}')
            continue
        if i > 0 and i % 1000 == 0:
            with open('data/writing-prompts/entity_grids.pkl', 'wb') as f:
                pickle.dump([grids, entities], f)

    return grids, entities


def find_rep(corefs, entity_idx):
    e_corefs = corefs[entity_idx]
    for coref in e_corefs:
        if coref['isRepresentativeMention']:
            return coref['text']



if __name__ == '__main__':
    # ptvsd.enable_attach(address=('0.0.0.0', 3700), redirect_output=True)
    # ptvsd.wait_for_attach()
    # setup_logger()
    opt = docopt(__doc__, version='0.0.1')
    opt = {k.lstrip('-'): v for k, v in opt.items()}

    grids, entities = extract_all_grids()

    logger.info('Saving grids to disk.')
    with open('data/writing-prompts/entity_grids.pkl', 'wb') as f:
        pickle.dump([grids, entities], f)
