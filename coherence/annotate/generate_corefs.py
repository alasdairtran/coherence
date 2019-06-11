"""CoreNLP Annotation

Make sure CoreNLP server is running. For example, in the terminal:

java -mx64g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 900000 -quiet

Usage:
    generate_corefs.py [--ann FODLER] [--corenlp URL]
    generate_corefs.py (-h | --help | --version)

Options:
    -h --help      Show this screen.
    --version      Show version.
    --ann FOLDER   The path of the output annotation folder
                   [default: data/writing-prompts/annotations]
    --corenlp URL  The URL of the CoreNLP server [default: http://localhost:9000].
"""


import json
import logging
import os
import time
from glob import glob

import requests
from docopt import docopt
from joblib import Parallel, delayed
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

logger = logging.getLogger(__name__)


neural_props = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,coref',
    'outputFormat': 'json',
    'coref.algorithm': 'neural',
    'ner.applyNumericClassifiers': False,
    'ner.applyFineGrained': False,
}

stats_props = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,coref',
    'outputFormat': 'json',
    'coref.algorithm': 'statistical',
    'ner.applyNumericClassifiers': False,
    'ner.applyFineGrained': False,
}


def annotate_story(text, name, corenlp_url, props=neural_props):
    out_path = os.path.join('data', 'writing-prompts',
                            'annotations', f'{name}.json')

    # Remove <newline>
    cleaned_text = []
    for token in text.split():
        if token != '<newline>':
            cleaned_text.append(token)
    cleaned_text = ' '.join(cleaned_text)

    if os.path.exists(out_path):
        return

    nlp = StanfordCoreNLP(corenlp_url)
    try:
        annotation = nlp.annotate(cleaned_text, properties=props)
    except requests.exceptions.ConnectionError as e:
        logger.error(f'Connection Error for {name}: {e}.')
        return

    if isinstance(annotation, str):
        logger.error(f'Error for {name}: {annotation}.')
        # Let's try a statistical approach
        if 'Error making document' in annotation and props['coref.algorithm'] == 'neural':
            logger.info(f'Switching to statistical coref for {name}')
            annotate_story(text, name, corenlp_url, props=stats_props)
        else:
            logger.info(f'Please check {name}')
            return
    else:
        with open(out_path, 'w') as f:
            json.dump(annotation, f)


if __name__ == '__main__':
    opt = docopt(__doc__, version='0.0.1')
    opt = {k.lstrip('-'): v for k, v in opt.items()}

    os.makedirs(opt['ann'], exist_ok=True)

    start = time.time()

    with open('./data/writing-prompts/train.wp_target') as f:
        with Parallel(n_jobs=-4, backend='threading') as parallel:
            annotations = parallel(delayed(annotate_story)(story, f'train-{i}', opt['corenlp'])
                                   for i, story in tqdm(enumerate(f)))

    with open('./data/writing-prompts/valid.wp_target') as f:
        with Parallel(n_jobs=-4, backend='threading') as parallel:
            annotations = parallel(delayed(annotate_story)(story, f'valid-{i}', opt['corenlp'])
                                   for i, story in tqdm(enumerate(f)))

    with open('./data/writing-prompts/test.wp_target') as f:
        with Parallel(n_jobs=-4, backend='threading') as parallel:
            annotations = parallel(delayed(annotate_story)(story, f'test-{i}', opt['corenlp'])
                                   for i, story in tqdm(enumerate(f)))

    duration = time.time() - start
    logger.info(f'Annotation of files took {duration}s.')
