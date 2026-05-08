import torch

from screening.module import MultiScreen
from screening.tokenizer import ABCDTokenizer


def test_init_multiscreen():
    num_blocks = 2
    hidden_dim = 64
    num_heads = 4
    window_threshold = 10.0

    tokenizer = ABCDTokenizer()
    vocab_size = tokenizer.vocab_size

    model = MultiScreen(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        window_threshold=window_threshold,
    )

    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        )

    print(logits.shape)
