from typing import Optional

from torch import Tensor, tensor
from torch.nn import Linear, LogSoftmax, Module

from utils.layers.embeddings import EmbeddingLayer
from utils.layers.encoders import EncoderLayer


class NeuralClassifier(Module):
    def __init__(self, components: dict[str, str], vocabularies: dict[str, dict],
                 embedding: EmbeddingLayer, encoder: EncoderLayer):
        super().__init__()
        self.components: dict[str, str] = components
        self.vocabularies: dict[str, dict] = vocabularies

        self.embedding: EmbeddingLayer = embedding
        self.encoder: EncoderLayer = encoder
        self.classification_layer: Linear = Linear(self.encoder.output_size, len(vocabularies["class_to_index"]))
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, batch_indices: Tensor, **batch_kwargs) -> Tensor:
        # Step 1: We take the input and convert it to its corresponding embeddings...
        embeddings: Tensor = self.embedding(batch_indices, **batch_kwargs["embedding"])   # (B, N) -> (B, N, E)

        # Step 2: We pass the embeddings through the encoder...
        encodings = self.encoder(embeddings, **batch_kwargs["encoder"])   # (B, N, E) -> (B, H)

        # Step 3: We pass the new representation through the final linear and log softmax layer,
        #   converting the representation to a set of log probabilities usable for classification.
        logits: Tensor = self.classification_layer(encodings)   # (B, H) -> (B, C)
        log_probabilities: Tensor = self.log_softmax(logits)   # (B, C) -> (B, C)
        return log_probabilities

    def prepare_classes(self, batch: list[str]) -> Tensor:
        indexed_classes: list[int] = [self.vocabularies["class_to_index"][item] for item in batch]
        class_tensor: Tensor = tensor(indexed_classes)
        return class_tensor

    def revert_classes(self, class_indices: list[int]) -> list[str]:
        named_classes: list[str] = [self.vocabularies["index_to_class"][index] for index in class_indices]
        return named_classes

    @staticmethod
    def prepare_distances(batch: list[float]) -> Optional[Tensor]:
        distance_tensor: Optional[Tensor] = tensor(batch) if len(batch) > 0 else None
        return distance_tensor
