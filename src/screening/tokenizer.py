import torch


class ABCDTokenizer:
    # - a~z
    # - A~Z
    # - 0~9
    # - space
    # - \n
    # - punctuation
    # - <|bos|><|eos|><|pad|>
    def __init__(self):
        self.vocab = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
            + [" ", "\n"]
            + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
            + ["<|bos|>", "<|eos|>", "<|pad|>"]
        )

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> torch.LongTensor:
        return torch.tensor(  # type: ignore
            [self.token_to_id[token] for token in text],
            dtype=torch.long,
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        return "".join(
            [self.id_to_token[int(token_id.item())] for token_id in token_ids]
        )

    def encode_batch(self, texts: list[str]) -> torch.LongTensor:
        max_length = max(len(text) for text in texts)
        batch_token_ids = torch.zeros((len(texts), max_length), dtype=torch.long)

        for i, text in enumerate(texts):
            token_ids = self.encode(text)
            batch_token_ids[i, : len(token_ids)] = token_ids

        return batch_token_ids  # type: ignore

    def decode_batch(self, batch_token_ids: torch.LongTensor) -> list[str]:
        return [self.decode(token_ids) for token_ids in batch_token_ids]
