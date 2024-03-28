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

## Use

The treebank-derived data points of the automatically-annotated datasets are licensed based on their datasets of origin.
Our datasets tag each point with their datasets of origin so that proper licensing can be taken into account.

We license the data points from each dataset as follows:
- [Perseus \[UD\]](https://github.com/UniversalDependencies/UD_Latin-Perseus) (Bamman and Crane, 2011): CC BY-NC-SA 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [PROIEL \[UD\]](https://github.com/UniversalDependencies/UD_Latin-PROIEL) (Haug *et al.*, 2009): CC BY-NC-SA 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [ITTB \[UD\]](https://github.com/UniversalDependencies/UD_Latin-ITTB) (Passarotti, 2019): CC BY-NC-SA 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [LLCT \[UD\]](https://github.com/UniversalDependencies/UD_Latin-LLCT) (Cecchini, Korkiakangas, and Passarotti, 2020a): CC BY-SA 4.0 International (https://creativecommons.org/licenses/by-sa/4.0/)
- [UDante \[UD\]](https://github.com/UniversalDependencies/UD_Latin-UDante) (Cecchini *et al.*, 2020b): CC BY-SA 4.0 International (https://creativecommons.org/licenses/by-sa/4.0/)
- [EvaLatin 2022](https://github.com/CIRCSE/LT4HALA/tree/master/2022) (Sprugnoli *et al.*, 2022): CC BY-NC-SA 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [Archimedes Latinus](https://github.com/mfantoli/ArchimedesLatinus) (Fantoli and de Lhoneux, 2022): CC BY-NC-SA 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/)

Please take caution when using our data that you adhere to the licensing applied by each set of data points, 
as we have not divided them into separate files based on licensing. 

## Contents

The repository contains a few main directories:
- `data`: a collection of data related to polarity detection; 
the data created through the study is given as a part of this repository in the subdirectories of `training/annotated`.
- `models`: a default location for models trained by `polarity_detector.py`.
- `predictions`: a collection of model predictions submitted to the EvaLatin 2024 shared task for polarity detection. 
The predictions are presented in the format expected by the scoring tool given by the organizers (see each `scorer` subdirectory) subdirectories 
as well as a format which resembles the original presentation of the data (see each `full` subdirectory).
- `resources`: a collection of external resources (usually pretrained models) for use in various tools (i.e., `gaussian_annotator.py` and `polarity_detector.py`).
- `results`: a default location for the results of training, validation, and testing produced by `polarity_detector.py`.
- `utils`: a collection of code organized into a variety of subdirectories and used by the variety of CLIs presented in the top level of this repository.

The repository mainly revolves around the CLIs present at its top level. 
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
applying SPhilBERTa embeddings (Riemenschneider and Frank, 2023b) with polarity coordinate features attached.

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
- **Embeddings**: Latin BERT (Bamman and Burns 2020), LaBERTa (Riemenschneider and Frank, 2023a), PhilBERTa (Riemenschneider and Frank, 2023a), SPhilBERTa (Riemenschneider and Frank, 2023b), mBERT (Devlin *et al.*, 2019), CANINE-C (Clark *et al.*, 2022), CANINE-S (Clark *et al.*, 2022)
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

```bibtex
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

For the datasets annotated automatically, refer to the following works:

```bibtex
@incollection{bammanAncientGreekLatin2011,
  title = {The {{Ancient Greek}} and {{Latin}} Dependency Treebanks},
  booktitle = {Language {{Technology}} for {{Cultural Heritage}}},
  author = {Bamman, David and Crane, Gregory},
  year = {2011},
  series = {Theory and {{Applications}} of {{Natural Language Processing}}},
  pages = {79--98},
  publisher = {{Springer Berlin Heidelberg}},
  address = {{Berlin, Heidelberg}},
  abstract = {This paper describes the development, composition, and several uses of the Ancient Greek and Latin Dependency Treebanks, large collections of Classical texts in which the syntactic, morphological and lexical information for each word is made explicit. To date, over 200 individuals from around the world have collaborated to annotate over 350,000 words, including the entirety of Homer's Iliad and Odyssey, Sophocles' Ajax, all of the extant works of Hesiod and Aeschylus, and selections from Caesar, Cicero, Jerome, Ovid, Petronius, Propertius, Sallust and Vergil. While perhaps the most straightforward value of such an annotated corpus for Classical philology is the morphosyntactic searching it makes possible, it also enables a large number of downstream tasks as well, such as inducing the syntactic behavior of lexemes and automatically identifying similar passages between texts.},
  copyright = {Springer-Verlag Berlin Heidelberg 2011},
  isbn = {3-642-20226-8},
  langid = {english},
  keywords = {Ancient Greek,dependency grammar,digital libraries,Latin,treebanks},
  url = {https://rdcu.be/dAKo0}
}

@inproceedings{haugComputationalLinguisticIssues2009,
  title = {Computational and Linguistic Issues in Designing a Syntactically Annotated Parallel Corpus of {{Indo-European}} Languages},
  booktitle = {Traitement Automatique Des Langues, Volume 50, Num{\'e}ro 2 : {{Langues}} Anciennes [{{Ancient}} Languages]},
  author = {Haug, Dag T. and J{\o}hndal, Marius L. and Eckhoff, Hanne M. and Welo, Eirik and Hertzenberg, Mari J. B. and M{\"u}th, Angelika},
  editor = {Denooz, Joseph and Rosmorduc, Serge},
  year = {2009},
  pages = {17--45},
  publisher = {{ATALA (Association pour le Traitement Automatique des Langues)}},
  address = {{France}},
  url = {https://aclanthology.org/2009.tal-2.2}
}

@incollection{passarottiProjectIndexThomisticus2019,
  title = {The {{Project}} of the {{Index Thomisticus Treebank}}},
  booktitle = {Digital {{Classical Philology}}},
  author = {Passarotti, Marco},
  year = {2019},
  volume = {10},
  pages = {299--320},
  publisher = {{De Gruyter}},
  address = {{Berlin, Boston}},
  copyright = {2019 Walter de Gruyter GmbH, Berlin/Munich/Boston},
  isbn = {3-11-059957-0},
  langid = {english},
  doi = {http://dx.doi.org/10.1515/9783110599572-017}
}

@inproceedings{cecchiniNewLatinTreebank2020a,
  title = {A New {{Latin}} Treebank for {{Universal Dependencies}}: {{Charters}} between {{Ancient Latin}} and {{Romance}} Languages},
  booktitle = {Proceedings of the 12th {{Language Resources}} and {{Evaluation Conference}}},
  author = {Cecchini, Flavio Massimiliano and Korkiakangas, Timo and Passarotti, Marco},
  year = {2020},
  month = may,
  pages = {933--942},
  publisher = {{European Language Resources Association}},
  address = {{Marseille, France}},
  abstract = {The present work introduces a new Latin treebank that follows the Universal Dependencies (UD) annotation standard. The treebank is obtained from the automated conversion of the Late Latin Charter Treebank 2 (LLCT2), originally in the Prague Dependency Treebank (PDT) style. As this treebank consists of Early Medieval legal documents, its language variety differs considerably from both the Classical and Medieval learned varieties prevalent in the other currently available UD Latin treebanks. Consequently, besides significant phenomena from the perspective of diachronic linguistics, this treebank also poses several challenging technical issues for the current and future syntactic annotation of Latin in the UD framework. Some of the most relevant cases are discussed in depth, with comparisons between the original PDT and the resulting UD annotations. Additionally, an overview of the UD-style structure of the treebank is given, and some diachronic aspects of the transition from Latin to Romance languages are highlighted.},
  isbn = {979-10-95546-34-4},
  langid = {english},
  url = {https://aclanthology.org/2020.lrec-1.117}
}

@inproceedings{cecchiniUDanteFirstSteps2020b,
  title = {{{UDante}}: {{First}} Steps towards the Universal Dependencies Treebank of {{Dante's}} {{Latin}} Works},
  booktitle = {Proceedings of the Seventh Italian Conference on Computational Linguistics, {{CLiC-it}} 2020, Bologna, Italy, March 1-3, 2021},
  author = {Cecchini, Flavio Massimiliano and Sprugnoli, Rachele and Moretti, Giovanni and Passarotti, Marco},
  editor = {Monti, Johanna and Dell'Orletta, Felice and Tamburini, Fabio},
  year = {2020},
  series = {{{CEUR}} Workshop Proceedings},
  volume = {2769},
  publisher = {{CEUR-WS.org}},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  timestamp = {Fri, 10 Mar 2023 16:22:17 +0100},
  url = {https://hdl.handle.net/11381/2913208}
}

@inproceedings{fantoliLinguisticAnnotationNeoLatin2022,
  title = {Linguistic Annotation of Neo-{{Latin}} Mathematical Texts: {{A}} Pilot-Study to Improve the Automatic Parsing of the {{Archimedes Latinus}}},
  booktitle = {Proceedings of the Second Workshop on Language Technologies for Historical and Ancient Languages},
  author = {Fantoli, Margherita and {de Lhoneux}, Miryam},
  editor = {Sprugnoli, Rachele and Passarotti, Marco},
  year = {2022},
  month = jun,
  pages = {129--134},
  publisher = {{European Language Resources Association}},
  address = {{Marseille, France}},
  abstract = {This paper describes the process of syntactically parsing the Latin translation by Jacopo da San Cassiano of the Greek mathematical work The Spirals of Archimedes. The Universal Dependencies formalism is adopted. First, we introduce the historical and linguistic importance of Jacopo da San Cassiano's translation. Subsequently, we describe the deep Biaffine parser used for this pilot study. In particular, we motivate the choice of using the technique of treebank embeddings in light of the characteristics of mathematical texts. The paper then details the process of creation of training and test data, by highlighting the most compelling linguistic features of the text and the choices implemented in the current version of the treebank. Finally, the results of the parsing are discussed in comparison to a baseline and the most prominent errors are discussed. Overall, the paper shows the added value of creating specific training data, and of using targeted strategies (as treebank embeddings) to exploit existing annotated corpora while preserving the features of one specific text when performing syntactic parsing.},
  url = {https://aclanthology.org/2022.lt4hala-1.18}
}

@inproceedings{sprugnoliOverviewEvaLatin2022,
  title = {Overview of the {{EvaLatin}} 2022 Evaluation Campaign},
  booktitle = {Proceedings of the Second Workshop on Language Technologies for Historical and Ancient Languages},
  author = {Sprugnoli, Rachele and Passarotti, Marco and Cecchini, Flavio Massimiliano and Fantoli, Margherita and Moretti, Giovanni},
  year = {2022},
  month = jun,
  pages = {183--188},
  publisher = {European Language Resources Association},
  address = {Marseille, France},
  abstract = {This paper describes the organization and the results of the second edition of EvaLatin, the campaign for the evaluation of Natural Language Processing tools for Latin. The three shared tasks proposed in EvaLatin 2022, i.,e.,Lemmatization, Part-of-Speech Tagging and Features Identification, are aimed to foster research in the field of language technologies for Classical languages. The shared dataset consists of texts mainly taken from the LASLA corpus. More specifically, the training set includes only prose texts of the Classical period, whereas the test set is organized in three sub-tasks: a Classical sub-task on a prose text of an author not included in the training data, a Cross-genre sub-task on poetic and scientific texts, and a Cross-time sub-task on a text of the 15th century. The results obtained by the participants for each task and sub-task are presented and discussed.},
  url = {https://aclanthology.org/2022.lt4hala-1.29}
}
```

For other works referenced above, see the following:

```bibtex
@misc{bammanLatinBERTContextual2020,
  title = {Latin {{BERT}}: {{A}} Contextual Language Model for Classical Philology},
  shorttitle = {Latin {{BERT}}},
  author = {Bamman, David and Burns, Patrick J.},
  year = {2020},
  month = sep,
  eprint = {2009.10053},
  urldate = {2020-09-27},
  abstract = {We present Latin BERT, a contextual language model for the Latin language, trained on 642.7 million words from a variety of sources spanning the Classical era to the 21st century. In a series of case studies, we illustrate the affordances of this language-specific model both for work in natural language processing for Latin and in using computational methods for traditional scholarship: we show that Latin BERT achieves a new state of the art for part-of-speech tagging on all three Universal Dependency datasets for Latin and can be used for predicting missing text (including critical emendations); we create a new dataset for assessing word sense disambiguation for Latin and demonstrate that Latin BERT outperforms static word embeddings; and we show that it can be used for semantically-informed search by querying contextual nearest neighbors. We publicly release trained models to help drive future work in this space.},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Computation and Language},
}

@inproceedings{riemenschneiderExploringLargeLanguage2023,
  title = {Exploring Large Language Models for Classical Philology},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: {{Long}} Papers)},
  author = {Riemenschneider, Frederick and Frank, Anette},
  editor = {Rogers, Anna and {Boyd-Graber}, Jordan and Okazaki, Naoaki},
  year = {2023},
  month = jul,
  pages = {15181--15199},
  publisher = {{Association for Computational Linguistics}},
  address = {{Toronto, Canada}},
  doi = {10.18653/v1/2023.acl-long.846},
  abstract = {Recent advances in NLP have led to the creation of powerful language models for many languages including Ancient Greek and Latin. While prior work on Classical languages unanimously uses BERT, in this work we create four language models for Ancient Greek that vary along two dimensions to study their versatility for tasks of interest for Classical languages: we explore (i) encoder-only and encoder-decoder architectures using RoBERTa and T5 as strong model types, and create for each of them (ii) a monolingual Ancient Greek and a multilingual instance that includes Latin and English. We evaluate all models on morphological and syntactic tasks, including lemmatization, which demonstrates the added value of T5's decoding abilities. We further define two probing tasks to investigate the knowledge acquired by models pre-trained on Classical texts. Our experiments provide the first benchmarking analysis of existing models of Ancient Greek. Results show that our models provide significant improvements over the SoTA. The systematic analysis of model types can inform future research in designing language models for Classical languages, including the development of novel generative tasks. We make all our models available as community resources, along with a large curated pre-training corpus for Ancient Greek, to support the creation of a larger, comparable model zoo for Classical Philology.}
}

@misc{riemenschneiderGraeciaCaptaFerum2023,
  title = {Graecia Capta Ferum Victorem Cepit. {{Detecting}} {{Latin}} Allusions to {{Ancient Greek}} Literature},
  author = {Riemenschneider, Frederick and Frank, Anette},
  year = {2023},
  eprint = {2308.12008},
  primaryclass = {cs.CL},
  archiveprefix = {arxiv},
  langid = {english}
}

@inproceedings{devlinBERTPretrainingDeep2019,
  title = {{{BERT}}: {{Pre-training}} of Deep Bidirectional Transformers for Language Understanding},
  booktitle = {Proceedings of the 2019 {{Conference}} of the {{North American Chapter}} of the {{Association}} for {{Computational Linguistics}}: {{Human Language Technologies}}, {{Volume}} 1 ({{Long}} and {{Short Papers}})},
  author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  year = {2019},
  month = jun,
  pages = {4171--4186},
  publisher = {{Association for Computational Linguistics}},
  address = {{Minneapolis, Minnesota}},
  doi = {10.18653/v1/N19-1423},
  abstract = {We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7\% (4.6\% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).},
}

@article{clarkCaninePretrainingEfficient2022,
  title = {Canine: {{Pre-training}} an {{Efficient Tokenization-Free Encoder}} for {{Language Representation}}},
  author = {Clark, Jonathan H. and Garrette, Dan and Turc, Iulia and Wieting, John},
  year = {2022},
  month = jan,
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {10},
  pages = {73--91},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00448},
  urldate = {2024-01-09},
  abstract = {Pipelined NLP systems have largely been superseded by end-to-end neural modeling, yet nearly all commonly used models still require an explicit tokenization step. While recent tokenization approaches based on data-derived subword lexicons are less brittle than manually engineered tokenizers, these techniques are not equally suited to all languages, and the use of any fixed vocabulary may limit a model's ability to adapt. In this paper, we present Canine, a neural encoder that operates directly on character sequences{\textemdash}without explicit tokenization or vocabulary{\textemdash}and a pre-training strategy that operates either directly on characters or optionally uses subwords as a soft inductive bias. To use its finer-grained input effectively and efficiently, Canine combines downsampling, which reduces the input sequence length, with a deep transformer stack, which encodes context. Canine outperforms a comparable mBert model by 5.7 F1 on TyDi QA, a challenging multilingual benchmark, despite having fewer model parameters.}
}
```
