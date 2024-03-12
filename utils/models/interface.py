from typing import Any, Type, Union

from utils.data.tokenizers.tokenizer import LatinLMTokenizer
from utils.layers.embeddings import EmbeddingLayer, get_embedding, NamedEmbedding
from utils.layers.encoders import EncoderLayer, get_encoder, NamedEncoder
from utils.models.bases import NamedArchitecture, NeuralArchitecture, NeuralClassifier, get_architecture


NamedComponent = Union[NamedEmbedding, NamedEncoder]


def build_model(named_architecture: NamedArchitecture, components: dict[str, NamedComponent],
                vocabularies: dict[str, Union[dict[str, int], list[str]]], tokenizer: LatinLMTokenizer,
                kwargs: dict[str, Any]) -> NeuralArchitecture:
    model_class: Type[NeuralArchitecture] = get_architecture(named_architecture)
    embedding: EmbeddingLayer = build_embedding(components["embedding"], tokenizer=tokenizer, **kwargs)
    encoder: EncoderLayer = build_encoder(components["encoder"], embedding, named_architecture, **kwargs)
    if named_architecture == NamedArchitecture.NEURAL_CLASSIFIER:
        model: NeuralClassifier = model_class(components, vocabularies, embedding, encoder)
    else:
        raise ValueError(f"The architecture <{named_architecture}> is not recognized.")

    return model


def build_embedding(named_embedding: NamedEmbedding, **kwargs) -> EmbeddingLayer:
    embedding_class: Type[EmbeddingLayer] = get_embedding(named_embedding)
    embedding_kwargs: dict[str, Any] = {
        "frozen_embeddings": kwargs["frozen_embeddings"],
        "pretrained_filepath": kwargs["pretrained_filepath"],
        "tokenizer": kwargs["tokenizer"]
    }

    embedding: EmbeddingLayer = embedding_class(named_embedding, **embedding_kwargs)
    return embedding


def build_encoder(named_encoder: NamedEncoder, embedding_layer: EmbeddingLayer, architecture: str, **kwargs) -> \
        EncoderLayer:
    encoder_class: Type[EncoderLayer] = get_encoder(named_encoder)
    encoder_kwargs: dict[str, Any] = {
        "input_size": embedding_layer.embedding_size,
        "layers": kwargs["layers"]
    }

    if named_encoder == NamedEncoder.IDENTITY:
        encoder_kwargs["hidden_size"] = embedding_layer.embedding_size
    elif named_encoder == NamedEncoder.LSTM:
        encoder_kwargs["bidirectional"] = kwargs["bidirectional"]
        encoder_kwargs["hidden_size"] = kwargs["hidden_size"]
    elif named_encoder == NamedEncoder.TRANSFORMER:
        encoder_kwargs["heads"] = kwargs["heads"]
        encoder_kwargs["hidden_size"] = kwargs["hidden_size"]
    else:
        raise ValueError(f"The encoder <{named_encoder}> is not currently recognized")

    if architecture == NamedArchitecture.NEURAL_CLASSIFIER:
        extractor_type: str = "single"
    else:
        raise ValueError(f"The architecture <{architecture}> is not recognized.")

    encoder_kwargs["extractor_type"] = extractor_type

    encoder: EncoderLayer = encoder_class(named_encoder, **encoder_kwargs)
    return encoder
