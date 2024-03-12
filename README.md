# EvaLatin 2024: Team Nostra Domina

**Authors**:
- Stephen Bothwell, Abigail Swenor, and David Chiang (University of Notre Dame)

**Maintainers**: Stephen Bothwell and Abigail Swenor

## Summary

This is a repository for the LT4HALA 2024 workshop paper, **"Improving Latin Polarity Detection through Semi-Supervised Data Augmentation."**
It includes:
- the automatically-annotated data generated as a part of the paper (see `data/polarity/training`) 
- the tools applied to generate and process that data, including the novel polarity coordinate clustering method (see `gaussian_annotator.py`, `polarity_annotator.py`, and `polarity_splitter.py`)
- the process used to train and tune neural networks on this data (see `polarity_detector.py` and `trial_generator.py`)
We describe the contents of this repository in more detail below, and we highlight our available CLIs as well as any additional information that may be helpful in reproducing our experiments.

## Contents

Below, we present information about the repository above. 
This mainly revolves around the CLIs present at the top level of this repository. 
The CLIs themselves are presented alongside descriptions of them and notes toward reproducing our work.
The below is further divided into two subsections regarding the initial data creation and dataset splitting (Data) 
and the hyperparameter tuning as well as the training and evaluation of neural networks (Modeling).

To reproduce our work, some data not innately included in our repository needs to be gathered. 
Links and references to all this data are present in `.gitkeep` files spread throughout the `data` and `resources` directories. 
An item's presence in a `.gitkeep` file indicates that it should be placed at that location. 
The `__init__.py` files under `utils/data/loaders` and `utils/layers/embeddings` provide indications on default filepaths for various elements of our data and pretrained embeddings, respectively.

### Data

In this section, we present our annotation tools (`gaussian_annotator.py`, `polarity_annotator.py`) as well as our data splitter (`polarity_splitter.py`). 

#### Gaussian Annotator

For our Gaussian annotator, we provide the CLI below:

```
>>> python gaussian_annotator.py -h
usage: gaussian_annotator.py [-h] [--components COMPONENTS] [--embedding-filepath EMBEDDING_FILEPATH] [--input-filepath INPUT_FILEPATH] [--lexicon-filepath LEXICON_FILEPATH]
                             [--output-filename OUTPUT_FILENAME] [--output-filepath OUTPUT_FILEPATH] [--report | --no-report] [--random-seed RANDOM_SEED]
                             [--random-seeds RANDOM_SEEDS] [--seed-filepath SEED_FILEPATH]

options:
  -h, --help            show this help message and exit
  --components COMPONENTS
                        number of distributions (i.e., classes) the Gaussian Mixture Model will incorporate
  --embedding-filepath EMBEDDING_FILEPATH
                        filepath to file or directory containing embedding data; currently only supports word2vec or SPhilBERTa
  --input-filepath INPUT_FILEPATH
                        filepath to directory containing data to annotate
  --lexicon-filepath LEXICON_FILEPATH
                        filepath to sentiment lexicon
  --output-filename OUTPUT_FILENAME
                        name of file where newly-annotated data will be stored; does not require file extension
  --output-filepath OUTPUT_FILEPATH
                        filepath to a directory where newly-annotated data will be stored
  --report, --no-report
                        a flag indicating whether the annotator should report numerical results regarding the data or not (default: False)
  --random-seed RANDOM_SEED
                        a flag indicating the random seed which controls the general procedure
  --random-seeds RANDOM_SEEDS
                        the number of random seeds used to generate initial random states for the Gaussian mixture Model
  --seed-filepath SEED_FILEPATH
                        filepath to labeled data used for training the Gaussian Mixture Model
```

The Gaussian annotator tool performs a hyperparameter search for the Gaussian Mixture Model (GMM) similar to the one performed during our experiments. 
We ran a total of 120 trials with the following hyperparameters and ranges:

- **Covariance Matrix Type**: diagonal (`diag`), full, spherical, tied
- **Initialization Method**: k-means, k-means++, random-from-data (`random_from_data`)

Each trial had ten different random initializations, and ten different random states were used to control this process. 
We raised the covariance regularization constant to .00001 to prevent the matrix from having negative values on the diagonal. 
All trials ran for up to 100 iterations. 
Due to size of the *Odes* dataset size, trials were both trained and evaluated on that set for their Macro-F1 score; 
the best GMM, having a tied covariance matrix and being initialized with the k-means algorithm, scored 0.37.

For the dataset produced with this work, we applied all defaults save that we set `--embedding-filepath` to point to `resources/sphilberta`, 
applying SPhilBERTa embeddings (Riemenschneider and Frank 2023) with polarity coordinate features attached.

#### Polarity Coordinate Annotator

For the polarity coordinate annotator, we provide the CLI below:

```
python polarity_annotator.py -h
usage: polarity_annotator.py [-h] [--input-filepath INPUT_FILEPATH] [--lexicon-filepath LEXICON_FILEPATH] [--output-filename OUTPUT_FILENAME] [--output-filepath OUTPUT_FILEPATH]
                             [--lexicon-sensitive | --no-lexicon-sensitive] [--report | --no-report]

options:
  -h, --help            show this help message and exit
  --input-filepath INPUT_FILEPATH
                        filepath to directory containing data to annotate
  --lexicon-filepath LEXICON_FILEPATH
                        filepath to sentiment lexicon
  --output-filename OUTPUT_FILENAME
                        name of file where newly-annotated data will be stored; does not require file extension
  --output-filepath OUTPUT_FILEPATH
                        filepath to a directory where newly-annotated data will be stored
  --lexicon-sensitive, --no-lexicon-sensitive
                        flag which determines whether only words in the sentiment lexicon are taken into account for computing polarity coordinates (default: False)
  --report, --no-report
                        a flag indicating whether the annotator should report numerical results regarding the data or not (default: False)
```

For the dataset produced with this work, we used the `--lexicon-sensitive` flag. All else was set to its defaults.

#### Polarity Splitter

For our data splitting, we used the CLI given below:

```
>>> python polarity_splitter.py -h        
usage: polarity_splitter.py [-h] [--input-file INPUT_FILE] [--names NAMES [NAMES ...]] [--output-directory OUTPUT_DIRECTORY] [--random-seed RANDOM_SEED]
                            [--ratios RATIOS [RATIOS ...]] [--strategy {random}]

options:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        filepath of the file to be partitioned into designated data splits
  --names NAMES [NAMES ...]
                        names of the data splits to be created; these must match the number of ratios given
  --output-directory OUTPUT_DIRECTORY
                        directory where data splits will be saved
  --random-seed RANDOM_SEED
                        a flag indicating the random seed which controls the general procedure
  --ratios RATIOS [RATIOS ...]
                        decimals representing the percentages of data composing each split; these must sum to 1, and the number of splits must match the number of names
  --strategy {random}   the algorithm used to create the data splits
```

We used the default values across the board for our data splitting procedures, 
save that we changed the file locations according to the annotation result that we were dealing with at the moment.

### Modeling

In this section, we present our neural network trainer and evaluator (`polarity_detector.py`) and our hyperparameter search script creator (`trial_generator.py`).

#### Polarity Detector

First, we provide the overall interface for the polarity detector. 
This interface hides most of its argument behind the `mode`---that is, the manner in which the detector is going to be used.
By doing this, certain arguments are locked to certain uses of the detector, and any incorrectly-applied arguments will throw errors upon being used.

```
>>> python polarity_detector.py -h
usage: polarity_detector.py [-h] {train,evaluate,predict} ...
                                                             
options:                                                     
  -h, --help            show this help message and exit

mode:
  {train,evaluate,predict}
```

Then, we provide the interface for the `train` mode:

```
>>> python polarity_detector.py train -h 
usage: polarity_detector.py train [-h] [--epochs EPOCHS] [--patience PATIENCE] [--bidirectional | --no-bidirectional] [--frozen-embeddings | --no-frozen-embeddings]
                                  [--heads [HEADS]] [--hidden-size HIDDEN_SIZE] [--layers [LAYERS]] [--lr [LR]] [--optimizer [OPTIMIZER]] [--output-filename OUTPUT_FILENAME]       
                                  [--output-location OUTPUT_LOCATION] [--training-filename [TRAINING_FILENAME]] [--training-interval [TRAINING_INTERVAL]]
                                  [--loss-function LOSS_FUNCTION] --dataset
                                  {NamedPolarityDataset.COORDINATE_TREEBANK,NamedPolarityDataset.GAUSSIAN_TREEBANK,NamedPolarityDataset.EVALATIN_2024_TEST,NamedPolarityDataset.HORACE_ODES}
                                  [--batch-size BATCH_SIZE] --embedding 
                                  {NamedEmbedding.CANINE_C,NamedEmbedding.CANINE_S,NamedEmbedding.LATIN_BERT,NamedEmbedding.LABERTA,NamedEmbedding.MULTILINGUAL_BERT,NamedEmbedding.PHILBERTA,NamedEmbedding.SPHILBERTA}
                                  --encoder {NamedEncoder.IDENTITY,NamedEncoder.LSTM,NamedEncoder.TRANSFORMER} [--evaluation-filename [EVALUATION_FILENAME]]
                                  [--model-location MODEL_LOCATION] [--model-name [MODEL_NAME]] [--pretrained-filepath PRETRAINED_FILEPATH] [--random-seed RANDOM_SEED]
                                  [--results-location [RESULTS_LOCATION]] [--tokenizer-filepath TOKENIZER_FILEPATH] [--tqdm | --no-tqdm]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       maximum number of epochs for training; must be included if patience is not
  --patience PATIENCE   maximum number of epochs without validation set improvement before early stopping; must be included if epochs is not
  --bidirectional, --no-bidirectional
                        flag determining whether an LSTM encoder should be monodirectional or bidirectional (default: False)
  --frozen-embeddings, --no-frozen-embeddings
                        a flag determining whether pretrained BERT-based embeddings will be subject to continued training or not (default: True)
  --heads [HEADS], --num-heads [HEADS]
                        number of attention heads per Transformer encoder layer
  --hidden-size HIDDEN_SIZE
                        size of the relevant encoder hidden state
  --layers [LAYERS], --num-layers [LAYERS]
                        number of encoder layers
  --lr [LR], --learning-rate [LR]
                        value representing the learning rate used by an optimizer during training
  --optimizer [OPTIMIZER]
                        optimization algorithm used during training
  --output-filename OUTPUT_FILENAME
                        name of file in which statistics regarding training will be saved; does not require file extension
  --output-location OUTPUT_LOCATION
                        directory in which statistics regarding training will be saved
  --training-filename [TRAINING_FILENAME]
                        filename prefix for outputted model training results
  --training-interval [TRAINING_INTERVAL]
                        interval of epochs after which the training set is used for evaluation
  --loss-function LOSS_FUNCTION
                        name of the loss function to be used during training
  --dataset {NamedPolarityDataset.COORDINATE_TREEBANK,NamedPolarityDataset.GAUSSIAN_TREEBANK,NamedPolarityDataset.EVALATIN_2024_TEST,NamedPolarityDataset.HORACE_ODES}
                        name of dataset to be used  
  --batch-size BATCH_SIZE
                        size of batches into which data is collated
  --embedding {NamedEmbedding.CANINE_C,NamedEmbedding.CANINE_S,NamedEmbedding.LATIN_BERT,NamedEmbedding.LABERTA,NamedEmbedding.MULTILINGUAL_BERT,NamedEmbedding.PHILBERTA,NamedEmbedding.SPHILBERTA}
                        type of embedding to be used in the neural architecture
  --encoder {NamedEncoder.IDENTITY,NamedEncoder.LSTM,NamedEncoder.TRANSFORMER}
                        type of encoder to be used in the neural architecture
  --evaluation-filename [EVALUATION_FILENAME]
                        filename prefix for validation, test, or prediction results
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name [MODEL_NAME]
                        filename prefix for a model at a designated location (see model-location)
  --pretrained-filepath PRETRAINED_FILEPATH
                        path to location of relevant pretrained BERT model; select 'auto' to use default filepath for given embedding
  --random-seed RANDOM_SEED
                        a flag indicating the random seed which controls the general procedure
  --results-location [RESULTS_LOCATION]
                        directory in which output files will be contained
  --tokenizer-filepath TOKENIZER_FILEPATH
                        filepath to subword tokenizer, if applicable
  --tqdm, --no-tqdm     flag for displaying tqdm-based iterations during processing (default: True)
```

Next, we provide the interface for the `evaluate` mode:

``` 
>>> python polarity_detector.py evaluate -h
usage: polarity_detector.py evaluate [-h] [--inference-split {training,validation,test}] --dataset
                                     {NamedPolarityDataset.COORDINATE_TREEBANK,NamedPolarityDataset.GAUSSIAN_TREEBANK,NamedPolarityDataset.EVALATIN_2024_TEST,NamedPolarityDataset.HORACE_ODES}
                                     [--batch-size BATCH_SIZE] --embedding 
                                     {NamedEmbedding.CANINE_C,NamedEmbedding.CANINE_S,NamedEmbedding.LATIN_BERT,NamedEmbedding.LABERTA,NamedEmbedding.MULTILINGUAL_BERT,NamedEmbedding.PHILBERTA,NamedEmbedding.SPHILBERTA}
                                     --encoder {NamedEncoder.IDENTITY,NamedEncoder.LSTM,NamedEncoder.TRANSFORMER} [--evaluation-filename [EVALUATION_FILENAME]]
                                     [--model-location MODEL_LOCATION] [--model-name [MODEL_NAME]] [--pretrained-filepath PRETRAINED_FILEPATH] [--random-seed RANDOM_SEED]
                                     [--results-location [RESULTS_LOCATION]] [--tokenizer-filepath TOKENIZER_FILEPATH] [--tqdm | --no-tqdm]

options:
  -h, --help            show this help message and exit
  --inference-split {training,validation,test}
                        name of data split to be used for evaluation
  --dataset {NamedPolarityDataset.COORDINATE_TREEBANK,NamedPolarityDataset.GAUSSIAN_TREEBANK,NamedPolarityDataset.EVALATIN_2024_TEST,NamedPolarityDataset.HORACE_ODES}
                        name of dataset to be used
  --batch-size BATCH_SIZE
                        size of batches into which data is collated
  --embedding {NamedEmbedding.CANINE_C,NamedEmbedding.CANINE_S,NamedEmbedding.LATIN_BERT,NamedEmbedding.LABERTA,NamedEmbedding.MULTILINGUAL_BERT,NamedEmbedding.PHILBERTA,NamedEmbedding.SPHILBERTA}
                        type of embedding to be used in the neural architecture
  --encoder {NamedEncoder.IDENTITY,NamedEncoder.LSTM,NamedEncoder.TRANSFORMER}
                        type of encoder to be used in the neural architecture
  --evaluation-filename [EVALUATION_FILENAME]
                        filename prefix for validation, test, or prediction results
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name [MODEL_NAME]
                        filename prefix for a model at a designated location (see model-location)
  --pretrained-filepath PRETRAINED_FILEPATH
                        path to location of relevant pretrained BERT model; select 'auto' to use default filepath for given embedding
  --random-seed RANDOM_SEED
                        a flag indicating the random seed which controls the general procedure
  --results-location [RESULTS_LOCATION]
                        directory in which output files will be contained
  --tokenizer-filepath TOKENIZER_FILEPATH
                        filepath to subword tokenizer, if applicable
  --tqdm, --no-tqdm     flag for displaying tqdm-based iterations during processing (default: True)
```

Finally, we provide the interface for the `predict` mode:

``` 
python polarity_detector.py predict -h  
usage: polarity_detector.py predict [-h] [--prediction-format {full,scorer}] [--inference-split {training,validation,test}] --dataset
                                    {NamedPolarityDataset.COORDINATE_TREEBANK,NamedPolarityDataset.GAUSSIAN_TREEBANK,NamedPolarityDataset.EVALATIN_2024_TEST,NamedPolarityDataset.HORACE_ODES}
                                    [--batch-size BATCH_SIZE] --embedding
                                    {NamedEmbedding.CANINE_C,NamedEmbedding.CANINE_S,NamedEmbedding.LATIN_BERT,NamedEmbedding.LABERTA,NamedEmbedding.MULTILINGUAL_BERT,NamedEmbedding.PHILBERTA,NamedEmbedding.SPHILBERTA}
                                    --encoder {NamedEncoder.IDENTITY,NamedEncoder.LSTM,NamedEncoder.TRANSFORMER} [--evaluation-filename [EVALUATION_FILENAME]]
                                    [--model-location MODEL_LOCATION] [--model-name [MODEL_NAME]] [--pretrained-filepath PRETRAINED_FILEPATH] [--random-seed RANDOM_SEED]
                                    [--results-location [RESULTS_LOCATION]] [--tokenizer-filepath TOKENIZER_FILEPATH] [--tqdm | --no-tqdm]

options:
  -h, --help            show this help message and exit
  --prediction-format {full,scorer}
                        format for predicted outputs to be written in
  --inference-split {training,validation,test}
                        name of data split to be used for evaluation
  --dataset {NamedPolarityDataset.COORDINATE_TREEBANK,NamedPolarityDataset.GAUSSIAN_TREEBANK,NamedPolarityDataset.EVALATIN_2024_TEST,NamedPolarityDataset.HORACE_ODES}
                        name of dataset to be used
  --batch-size BATCH_SIZE
                        size of batches into which data is collated
  --embedding {NamedEmbedding.CANINE_C,NamedEmbedding.CANINE_S,NamedEmbedding.LATIN_BERT,NamedEmbedding.LABERTA,NamedEmbedding.MULTILINGUAL_BERT,NamedEmbedding.PHILBERTA,NamedEmbedding.SPHILBERTA}
                        type of embedding to be used in the neural architecture
  --encoder {NamedEncoder.IDENTITY,NamedEncoder.LSTM,NamedEncoder.TRANSFORMER}
                        type of encoder to be used in the neural architecture
  --evaluation-filename [EVALUATION_FILENAME]
                        filename prefix for validation, test, or prediction results
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name [MODEL_NAME]
                        filename prefix for a model at a designated location (see model-location)
  --pretrained-filepath PRETRAINED_FILEPATH
                        path to location of relevant pretrained BERT model; select 'auto' to use default filepath for given embedding
  --random-seed RANDOM_SEED
                        a flag indicating the random seed which controls the general procedure
  --results-location [RESULTS_LOCATION]
                        directory in which output files will be contained
  --tokenizer-filepath TOKENIZER_FILEPATH
                        filepath to subword tokenizer, if applicable
  --tqdm, --no-tqdm     flag for displaying tqdm-based iterations during processing (default: True)
```

#### Trial Generator

The interface to our hyperparameter search trial generator is given below. 
Currently, it only supports performing a random search. It assures that duplicate trials are not chosen, 
but it otherwise does not intervene on the trial generation in any way.

```
>>> python trial_generator.py -h
usage: trial_generator.py [-h] [--command-filepath COMMAND_FILEPATH] [--command-format {text,bash}] [--model-location MODEL_LOCATION] [--model-name MODEL_NAME]
                          [--results-location RESULTS_LOCATION] [--training-filename TRAINING_FILENAME] [--validation-filename VALIDATION_FILENAME]            
                          [--test-filename TEST_FILENAME] [--training-interval TRAINING_INTERVAL] [--inference-split {train,validation,test}] [--seed SEED]    
                          [--specified [SPECIFIED ...]] [--trials TRIALS] [--trial-start-offset TRIAL_START_OFFSET]
                          [--varied [{batch-size,bidirectional,dataset,embedding,encoder,epochs,frozen-embeddings,heads,hidden-size,layers,learning-rate,loss-function,patience,random-seed} ...]]

options:
  -h, --help            show this help message and exit
  --command-filepath COMMAND_FILEPATH
                        filepath prefix for outputted trial files
  --command-format {text,bash}
                        filetype for outputted trial files
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name MODEL_NAME
                        filename prefix for a model at a designated location (see model-location)
  --results-location RESULTS_LOCATION
                        directory in which output files will be contained
  --training-filename TRAINING_FILENAME
                        filename prefix for outputted model training results
  --validation-filename VALIDATION_FILENAME
                        filename prefix for outputted model validation results
  --test-filename TEST_FILENAME
                        filename prefix for outputted model test results
  --training-interval TRAINING_INTERVAL
                        interval of epochs after which the training set is used for evaluation
  --inference-split {train,validation,test}
                        name of data split to be used for evaluation
  --seed SEED           a flag indicating the random seed which controls the general procedure
  --specified [SPECIFIED ...]
                        hyperparameter names and values which will be fixed across all trials
  --trials TRIALS       number of hyperparameter trials to generate
  --trial-start-offset TRIAL_START_OFFSET
                        starting number for hyperparameter trial generation
  --varied [{batch-size,bidirectional,dataset,embedding,encoder,epochs,frozen-embeddings,heads,hidden-size,layers,learning-rate,loss-function,patience,random-seed} ...]
                        hyperparameters which will be varied across all trials

```

For our experiments, we ran sets of 4 trials across a variety of scenarios. In particular, we ran variations on the following model components:
- **Embeddings**: Latin BERT, LaBERTa, PhilBERTa, SPhilBERTa, mBERT, CANINE-C, CANINE-S
- **Encoders**: Identity, BiLSTM, Transformer
- **Loss Functions**: Cross-Entropy, Gold Distance-Weighted Cross Entropy
- **Datasets**: Polarity Coordinate, Gaussian

Note that not all combinations were used. Namely, SPhilBERTa, capturing sentence-level rather than word-level representations, 
uses only the Identity encoder. Moreover, the same embedding was not applied with the Gaussian dataset, as it was used to train the GMM which generated that dataset. 
Finally, the Gold Distance-Weighted Cross Entropy loss is only used with the Polarity Coordinate data, as the Gaussian data has no such measure of "distance" to use.

During these trials, we varied the learning rate, the hidden size (when applicable), and number of encoder layers (when applicable). 
Their ranges were as follows:
- **Hidden Size**: {64, 96, 128, 192, 256, 384, 512, 768, 1024}
- **Layers**: {1, 2, 3, 4}
- **Learning Rate**: {1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 9e-05, 1e-04, 2e-04, 3e-04, 4e-04, 5e-04, 6e-04, 7e-04, 8e-04, 9e-04, 1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02}

These ranges were selected based upon a mixture of previous experience and in consulting related work. 
To elaborate further, the hidden size range consists of all powers of 2 from 2\*\*6 and 2\*\*10 alongside each average of each pair of successively larger powers of 2. 
The learning rate range contains progressively longer intervals between learning rates, starting with 1e-05 to 9e-05 (in steps of 1e-05), following that with 1e-04 to 9e-04 (in steps of 1e-04), following that with 1e-03 to 9e-03 (in steps of 1e-03), and concluding with 1e-02.
This range is an attempt to keep the number of possible learning rates low while covering a decent range and including ones found in related works.

Regarding the learning rate range, there was a mild mistake in the code which caused 1e-04 and 1e-03 appeared twice. 
This made them slightly more likely to appear than the other values.
To allow our experiments to be reduplicated exactly, we did not correct this. 
However, extensions of this code should be aware of it so that they may correct or incorporate it as they wish.

## Contributing

This repository contains code relating to our submission to EvaLatin 2024, an evaluation campaign at the [LT4HALA](https://circse.github.io/LT4HALA/) workshop. 
The code was altered before submission to promote usability, but it is possible that bugs were introduced in the interim. 
If you experience issues in using this code or would like more instructions in reproducing our results, please feel free to submit an issue regarding this.

We do not intend to heavily maintain this code, as it is meant to represent our paper at its time of publication. 
Exceptions may be made if warranted (*e.g.*, there is a bug which prevents the code from being correctly run), 
and we are happy to provide clarifications or assistance in reproducing our results.

## Citations

To cite this repository, please use the following citation:

```
@inproceedings{bothwellImprovingLatinPolarity2024,
  title = {Improving {{Latin}} Polarity Detection through Semi-Supervised Data Augmentation},
  booktitle = {Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages},
  author = {Bothwell, Stephen and Swenor, Abigail and Chiang, David},
  year = {2024},
  month = may,
  publisher = {European Language Resources Association},
  address = {Turin, Italy},
  abstract = {This paper describes submissions from the team Nostra Domina to the EvaLatin 2024 shared task of emotion polarity detection. Given the low-resource environment of Latin and the complexity of sentiment in rhetorical genres like poetry, we augmented the available sentiment data through semi-supervised polarity annotation. We present two methods for doing so on the basis of the \$k\$-means algorithm, and we employ a variety of Latin large language models (LLMs) in a neural architecture to better capture the underlying contextual sentiment representations. Our approach achieved the second best Macro-F{\textbackslash}textsubscript\{1\} score on the shared task's test set.},
  langid = {english},
  annotation = {To appear.}
}
```

For other works referenced above, see the following:
```
@misc{riemenschneiderGraeciaCaptaFerum2023,
  title = {Graecia Capta Ferum Victorem Cepit. {{Detecting}} Latin Allusions to Ancient Greek Literature},
  author = {Riemenschneider, Frederick and Frank, Anette},
  year = {2023},
  eprint = {2308.12008},
  primaryclass = {cs.CL},
  archiveprefix = {arxiv},
  langid = {english}
}
```