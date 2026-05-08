from abc import ABC, abstractmethod
from typing import NamedTuple

import torch


class TokenizerOutput(NamedTuple):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor


class TokenizerProtocol(ABC):
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> TokenizerOutput:
        pass

    @abstractmethod
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        pass

    @abstractmethod
    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = False
    ) -> TokenizerOutput:
        pass

    @abstractmethod
    def decode_batch(
        self, batch_token_ids: torch.LongTensor, skip_special_tokens: bool = False
    ) -> list[str]:
        pass

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> TokenizerOutput:
        return self.encode(text, add_special_tokens=add_special_tokens)


class ABCDTokenizer(TokenizerProtocol):
    # - a~z
    # - A~Z
    # - 0~9
    # - space
    # - \n
    # - punctuation
    # - <|bos|><|eos|><|pad|>
    def __init__(self):
        self.special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>"]
        self.vocab = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
            + [" ", "\n"]
            + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
            + self.special_tokens
        )

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.pad_token = "<|pad|>"
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.pad_token_id = self.token_to_id[self.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def remove_special_tokens(self, text: str) -> str:
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> TokenizerOutput:
        if add_special_tokens:
            text = self.bos_token + text + self.eos_token
        input_ids = torch.tensor(
            [self.token_to_id[token] for token in text],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        return TokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        text = "".join(
            [self.id_to_token[int(token_id.item())] for token_id in token_ids]
        )
        if skip_special_tokens:
            text = self.remove_special_tokens(text)
        return text

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = False
    ) -> TokenizerOutput:
        max_length = max(len(text) for text in texts)
        batch_input_ids = torch.full(
            (len(texts), max_length), fill_value=self.pad_token_id, dtype=torch.long
        )
        batch_attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)

        for i, text in enumerate(texts):
            token_output = self.encode(text, add_special_tokens=add_special_tokens)
            batch_input_ids[i, : len(token_output.input_ids)] = token_output.input_ids
            batch_attention_mask[i, : len(token_output.attention_mask)] = (
                token_output.attention_mask
            )

        return TokenizerOutput(
            input_ids=batch_input_ids,  # type: ignore
            attention_mask=batch_attention_mask,  # type: ignore
        )

    def decode_batch(
        self, batch_token_ids: torch.LongTensor, skip_special_tokens: bool = False
    ) -> list[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]
