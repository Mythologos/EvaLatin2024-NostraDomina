from torch import clamp, sum, Tensor

from utils.layers.embeddings import LatinLMEmbedding


class LatinSentenceLMEmbedding(LatinLMEmbedding):
    def __init__(self, component: str, **kwargs):
        super().__init__(component, **kwargs)

    def forward(self, batch: Tensor, **kwargs) -> Tensor:   # (B, N) -> (B, N, E).
        token_embeddings: Tensor = self.lm(input_ids=batch, attention_mask=kwargs["attention_mask"]).last_hidden_state
        sentence_embeddings: Tensor = self._mean_pooling(token_embeddings, attention_mask=kwargs["attention_mask"])
        return sentence_embeddings

    # This function is from the Sentence Transformers documentation.
    @staticmethod
    def _mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        # The first element of `embeddings` contains all token embeddings.
        expanded_mask: Tensor = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed_embeddings: Tensor = sum(token_embeddings * expanded_mask, keepdim=True, dim=1)
        summed_mask: Tensor = clamp(expanded_mask.sum(dim=1, keepdim=True), min=1e-9)
        sentence_embeddings: Tensor = summed_embeddings / summed_mask
        return sentence_embeddings
