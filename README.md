# Text Coherence Experiments

This repo contains a collection of experiments related to text coherence.

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
```
