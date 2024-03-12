# EvaLatin 2024: Team Nostra Domina

**Authors**:
- Stephen Bothwell, Abigail Swenor, and David Chiang (University of Notre Dame)

**Maintainers**: Stephen Bothwell and Abigail Swenor

## Summary

...

## Contents

...

### Data

#### Gaussian Annotator

...

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

#### Polarity Coordinate Annotator

...

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

#### Polarity Splitter

...

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

### Modeling

#### Polarity Detector

...

```
>>> python polarity_detector.py -h
usage: polarity_detector.py [-h] {train,evaluate,predict} ...
                                                             
options:                                                     
  -h, --help            show this help message and exit

mode:
  {train,evaluate,predict}
```

training:

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

evaluation:

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

prediction:

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

...

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
...
```