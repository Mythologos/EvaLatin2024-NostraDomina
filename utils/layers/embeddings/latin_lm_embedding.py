from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from utils.layers.embeddings import EmbeddingLayer


class LatinLMEmbedding(EmbeddingLayer):
    def __init__(self, component: str, **kwargs):
        bert_config: PretrainedConfig = AutoConfig.from_pretrained(kwargs["pretrained_filepath"])
        super().__init__(component, bert_config.hidden_size)

        self.lm: PreTrainedModel = AutoModel.from_pretrained(kwargs["pretrained_filepath"])
        self.frozen_embeddings: bool = kwargs["frozen_embeddings"]

        if self.frozen_embeddings is True:
            for parameter in self.lm.parameters():
                parameter.requires_grad = False

    def forward(self, batch: Tensor, **kwargs) -> Tensor:   # (B, N) -> (B, N, E).
        embeddings: Tensor = self.lm(input_ids=batch, attention_mask=kwargs["attention_mask"]).last_hidden_state
        return embeddings
