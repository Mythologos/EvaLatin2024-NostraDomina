from utils.constants import NamedEnum


class AnnotatorMessage(NamedEnum):
    INPUT_FILEPATH: str = "filepath to directory containing data to annotate"
    LEXICON_FILEPATH: str = "filepath to sentiment lexicon"
    OUTPUT_FILENAME: str = "name of file where newly-annotated data will be stored; does not require file extension"
    OUTPUT_FILEPATH: str = "filepath to a directory where newly-annotated data will be stored"
    REPORT: str = "a flag indicating whether the annotator should report numerical results regarding the data or not"


class DetectorMessage(NamedEnum):
    BATCH_SIZE: str = "size of batches into which data is collated"
    BIDIRECTIONAL: str = "flag determining whether an LSTM encoder should be monodirectional or bidirectional"
    DATASET: str = "name of dataset to be used"
    EMBEDDING: str = "type of embedding to be used in the neural architecture"
    ENCODER: str = "type of encoder to be used in the neural architecture"
    EPOCHS: str = "maximum number of epochs for training; must be included if patience is not"
    EVALUATION_FILENAME: str = "filename prefix for validation, test, or prediction results"
    HEADS: str = "number of attention heads per Transformer encoder layer"
    FROZEN_EMBEDDINGS: str = "a flag determining whether pretrained BERT-based embeddings " \
                             "will be subject to continued training or not"
    HIDDEN_SIZE: str = "size of the relevant encoder hidden state"
    LAYERS: str = "number of encoder layers"
    LEARNING_RATE: str = "value representing the learning rate used by an optimizer during training"
    LOSS_FUNCTION: str = "name of the loss function to be used during training"
    MODE: str = "indication of whether a model is being subject to training, evaluation, or prediction"
    OPTIMIZER: str = "optimization algorithm used during training"
    OUTPUT_LOCATION: str = "directory in which statistics regarding training will be saved"
    OUTPUT_FILENAME: str = "name of file in which statistics regarding training will be saved; " \
                           "does not require file extension"
    PATIENCE: str = "maximum number of epochs without validation set improvement before early stopping; " \
                    "must be included if epochs is not"
    PREDICTION_FORMAT: str = "format for predicted outputs to be written in"
    PRETRAINED_FILEPATH: str = "path to location of relevant pretrained BERT model; " \
                               "select 'auto' to use default filepath for given embedding"
    TOKENIZER_FILEPATH: str = "filepath to subword tokenizer, if applicable"
    TQDM: str = "flag for displaying tqdm-based iterations during processing"


class GaussianMessage(NamedEnum):
    COMPONENTS: str = "number of distributions (i.e., classes) the Gaussian Mixture Model will incorporate"
    EMBEDDING_FILEPATH: str = "filepath to file or directory containing embedding data; " \
                              "currently only supports word2vec or SPhilBERTa"
    SEED_FILEPATH: str = "filepath to labeled data used for training the Gaussian Mixture Model"
    RANDOM_SEEDS: str = "the number of random seeds used to generate initial random states " \
                        "for the Gaussian mixture Model"


class GeneralMessage(NamedEnum):
    INFERENCE_SPLIT: str = "name of data split to be used for evaluation"
    MODEL_LOCATION: str = "directory in which a model is or will be contained"
    MODEL_NAME: str = "filename prefix for a model at a designated location (see model-location)"
    RANDOM_SEED: str = "a flag indicating the random seed which controls the general procedure"
    RESULTS_LOCATION: str = "directory in which output files will be contained"
    TRAINING_FILENAME: str = "filename prefix for outputted model training results"
    TRAINING_INTERVAL: str = "interval of epochs after which the training set is used for evaluation"


class HyperparameterMessage(NamedEnum):
    COMMAND_FILEPATH: str = "filepath prefix for outputted trial files"
    COMMAND_FORMAT: str = "filetype for outputted trial files"
    INFERENCE_SPLIT: str = "name of split to be used for evaluation"
    SPECIFIED: str = "hyperparameter names and values which will be fixed across all trials"
    TEST_FILENAME: str = "filename prefix for outputted model test results"
    TRIALS: str = "number of hyperparameter trials to generate"
    TRIAL_OFFSET: str = "starting number for hyperparameter trial generation"
    VALIDATION_FILENAME: str = "filename prefix for outputted model validation results"
    VARIED: str = "hyperparameters which will be varied across all trials"


class PCMessage(NamedEnum):
    LEXICON_SENSITIVE: str = "flag which determines whether only words in the sentiment lexicon " \
                             "are taken into account for computing polarity coordinates"


class SplitterMessage(NamedEnum):
    INPUT_FILE: str = "filepath of the file to be partitioned into designated data splits"
    NAMES: str = "names of the data splits to be created; these must match the number of ratios given"
    OUTPUT_DIRECTORY: str = "directory where data splits will be saved"
    RATIOS: str = "decimals representing the percentages of data composing each split; these must sum to 1, " \
                  "and the number of splits must match the number of names"
    STRATEGY: str = "the algorithm used to create the data splits"
