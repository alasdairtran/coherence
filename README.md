# Text Coherence Experiments

This repo contains a collection of experiments related to text coherence.
In particular we explored whether neural entity grid models can be used
to give a passage of text a meaningful coherence score. The model is trained
using ranking loss, in which a entity grid of the original text is ranked
higher than a grid with some noise added to it. An example of noise could be
randomly moving an entity to a different place in the text, or adding a random
entity mention.

## Setting Up

```sh
# Install dependencies
pip install docopt StanfordDependencies pycorenlp tqdm allenlp ptvsd

# To get half-precision support, install apex. GPUs with device capability at
# least 7 (e.g. Titan V) have support for half-precision training.
git clone https://github.com/nvidia/apex
cd apex && python setup.py install --cuda_ext --cpp_ext

# Install BERT server
pip install -U tf-nightly-gpu bert-serving-server bert-serving-client
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip

# Download Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/index.html
# Ensure that stanford-corenlp-3.9.2.jar is in PATH.
CORENLP_DIR=$HOME/lib/corenlp # Change this to directory where CoreNLP is installed
for file in `find $CORENLP_DIR -name "*.jar"`; do
    export CLASSPATH="${CLASSPATH:+${CLASSPATH}:}`realpath $file`"; done

# Install this coherence repo
python setup.py develop
```

## Annotation and Training

```sh
# Remember to run all of the commands below from the root project directory,
# i.e. the directory containing this README file.

# Download the Reddit writing prompt dataset
./download

# Ensure that the CoreNLP server is running
java -mx64g edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -port 9000 \
    -timeout 900000 \
    -quiet

# Annotnate reddit data (this could take 2 days). This will generate annotation
# files in JSON format inside data/writing-prompts/annotations/
python -m coherence.annotate.generate_corefs \
    --ann data/writing-prompts/annotations \
    --corenlp http://localhost:9000

# Generate entity grids. This will create data/writing-prompts/entity_grids.pkl
python -m coherence.annotate.generate_grids

# Ensure that the BERT server is running
bert-serving-start \
    -model_dir cased_L-12_H-768_A-12 \
    -num_worker 1 \
    -device_map 0 \
    -gpu_memory_fraction 0.1 \
    -max_seq_len 128 \
    -max_batch_size 64

# Get BERT embeddings. This will create data/writing-prompts/entity_grids_with_embeds.pkl
python -m coherence.annotate.generate_grid_berts

# Training entity grid model
CUDA_VISIBLE_DEVICES=0 python -m coherence.commands train experiments/entity_grid_ranking/config.yaml -fb

# Evaluate the model
CUDA_VISIBLE_DEVICES=0 python -m coherence.commands evaluate experiments/entity_grid_ranking/model.tar.gz
```

## Results

In the table below, each row is a possible type of noise that we could add to
the grid. The second column shows the percentage of grids in each category that
are ranked lower than the original grid. All examples are taken from the
validation set. These results match what we would like to happen. That is,
the more noise we add to the grid, the lower the score it would get, compared
to the original grid.

|                                     | % that ranked lower (validation) |
| ----------------------------------- | -------------------------------- |
| Add 5 mentions of existing entities | 90%                              |
| Add a mention of an existing entity | 73%                              |
| Swap two sentences                  | 65%                              |
| Turn off one random entry           | 58%                              |
| Turn off two random entries         | 63%                              |
| Noisy Grid (5% randomised)          | 75%                              |
| Noisy Grid (10% randomised)         | 87%                              |
| Noisy Grid (20% randomised)         | 93%                              |
| Noisy Grid (30% randomised)         | 96%                              |
| Randomly Shuffled Sentences         | 92%                              |
