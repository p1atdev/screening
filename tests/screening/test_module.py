import math

import torch

from screening.modules import (
    GatedScreening,
    apply_mipe,
    causal_softmask,
    mipe_rotation,
    compute_freqs_cis,
    screening,
    tanh_norm,
    trim_similarity,
    unit_length_norm,
)
from screening.models import MultiScreenForCausalLM
from screening.tokenizer import ABCDTokenizer


def test_unit_length_norm_handles_zero_vectors():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    normalized = unit_length_norm(x)

    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized[0], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(normalized[1], torch.zeros(2))


def test_tanh_norm_handles_zero_and_bounds_norm():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    normalized = tanh_norm(x)

    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized[0], torch.zeros(2))
    torch.testing.assert_close(normalized[1].norm(), torch.tanh(torch.tensor(5.0)))


def test_trim_similarity_uses_per_head_acceptance():
    similarity = torch.tensor([[[[1.0, 0.5, 0.0]], [[1.0, 0.5, 0.0]]]])
    acceptance = torch.tensor([0.5, 1.0])

    relevance = trim_similarity(similarity, acceptance)

    expected = torch.tensor([[[[1.0, 0.0, 0.0]], [[1.0, 0.25, 0.0]]]])
    torch.testing.assert_close(relevance, expected)


def test_mipe_rotation_is_disabled_at_threshold_and_above():
    position_ids = torch.tensor([[0, 1, 2]])
    window = torch.tensor([4.0, 8.0, 16.0])

    rotation = mipe_rotation(
        position_ids=position_ids,
        window=window,
        window_threshold=8.0,
    )

    torch.testing.assert_close(
        rotation[0, 0],
        torch.tensor([0.0, math.pi / 8, math.pi / 4]),
    )
    torch.testing.assert_close(rotation[0, 1], torch.zeros(3))
    torch.testing.assert_close(rotation[0, 2], torch.zeros(3))


def test_apply_mipe_rotates_only_first_two_dimensions():
    sequence = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 5.0, 2.0, 3.0],
                    [0.0, 1.0, 6.0, 4.0, 1.0],
                ]
            ]
        ]
    )  # [batch_size=1, num_heads=1, seq_len=2, head_dim=5]
    position_ids = torch.tensor([[0, 1]])

    freqs_cis = compute_freqs_cis(
        position_ids=position_ids,
        window=torch.tensor([4.0]),
        window_threshold=8.0,
    )

    encoded = apply_mipe(
        sequence=sequence,
        freqs_cis=freqs_cis,
    )

    torch.testing.assert_close(
        encoded[0, 0, 0], torch.tensor([1.0, 0.0, 5.0, 2.0, 3.0])
    )
    torch.testing.assert_close(
        encoded[0, 0, 1],
        torch.tensor(
            [
                -math.sin(math.pi / 8),
                math.cos(math.pi / 8),
                6.0,
                4.0,
                1.0,
            ]
        ),
    )


def test_apply_mipe_rotates_2d():
    sequence = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 5.0, 2.0, 3.0],
                    [0.0, 1.0, 6.0, 4.0, 1.0],
                    [0.0, 0.0, 7.0, 8.0, 9.0],
                    [0.0, 2.0, 8.0, 4.0, 3.0],
                ]
            ]
        ]
    )  # [batch_size=1, num_heads=1, seq_len=4, head_dim=5]
    position_ids = torch.tensor(
        [
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
        ]
    )  # [batch_size=1, seq_len=4, num_axes=2]

    freqs_cis = compute_freqs_cis(
        position_ids=position_ids,
        window=torch.tensor([4.0]),
        window_threshold=8.0,
    )

    encoded = apply_mipe(
        sequence=sequence,
        freqs_cis=freqs_cis,
    )

    torch.testing.assert_close(
        encoded[0, 0, 0], torch.tensor([1.0, 0.0, 5.0, 2.0, 3.0])
    )
    torch.testing.assert_close(
        encoded[0, 0, 3],
        torch.tensor(
            [
                -2 * math.sin(math.pi / 8),
                2 * math.cos(math.pi / 8),
                8 * math.cos(math.pi / 8) - 4 * math.sin(math.pi / 8),
                8 * math.sin(math.pi / 8) + 4 * math.cos(math.pi / 8),
                3.0,
            ]
        ),
    )
    assert sequence.shape == encoded.shape


def test_causal_softmask_masks_future_and_out_of_window_positions():
    mask = causal_softmask(
        position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        window=torch.tensor([2.0]),
    )

    expected = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.0, 0.5, 1.0],
                ]
            ]
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_causal_softmask_accepts_singleton_axis_position_ids():
    position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    singleton_axis_position_ids = position_ids.unsqueeze(-1)

    mask = causal_softmask(
        position_ids=position_ids,
        window=torch.tensor([2.0]),
    )
    singleton_axis_mask = causal_softmask(
        position_ids=singleton_axis_position_ids,
        window=torch.tensor([2.0]),
    )

    torch.testing.assert_close(singleton_axis_mask, mask)


def test_screening_uses_default_causal_positions_without_mipe_position_ids():
    query = torch.zeros(1, 1, 2, 2)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    query[..., 0] = 1.0
    key[..., 0] = 1.0
    value[0, 0, 0] = torch.tensor([1.0, 0.0])
    value[0, 0, 1] = torch.tensor([0.0, 1.0])

    screened = screening(
        query=query,
        key=key,
        value=value,
        window=torch.tensor([10.0]),
        window_threshold=1.0,
        acceptance=torch.tensor([1.0]),
        position_ids=None,
    )

    torch.testing.assert_close(screened[0, 0, 0, 1], torch.tensor(0.0))


def test_screening_attention_mask_removes_masked_key_contribution():
    query = torch.zeros(1, 1, 3, 2)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    query[..., 0] = 1.0
    key[..., 0] = 1.0
    value[0, 0, 0] = torch.tensor([1.0, 0.0])
    value[0, 0, 1] = torch.tensor([0.0, 1.0])
    value[0, 0, 2] = torch.tensor([0.0, 1.0])

    unmasked = screening(
        query=query,
        key=key,
        value=value,
        position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        window=torch.tensor([10.0]),
        window_threshold=1.0,
        acceptance=torch.tensor([1.0]),
    )
    masked = screening(
        query=query,
        key=key,
        value=value,
        position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        window=torch.tensor([10.0]),
        window_threshold=1.0,
        acceptance=torch.tensor([1.0]),
        attention_mask=torch.tensor([[1, 0, 1]], dtype=torch.bool),
    )

    assert unmasked[0, 0, 1, 1] > 0
    torch.testing.assert_close(masked[0, 0, 1, 1], torch.tensor(0.0))


def test_gated_screening_traces_score_matrix_and_gate_outputs():
    layer = GatedScreening(
        hidden_dim=8,
        num_heads=2,
        window_threshold=8.0,
        is_causal=False,
    )
    layer.set_trace_screening(True)
    hidden_states = torch.randn(1, 4, 8)
    position_ids = torch.tensor(
        [
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        ]
    )

    output = layer(hidden_states, position_ids=position_ids)

    assert output.shape == hidden_states.shape
    assert set(layer.screening_trace) == {"score", "before_gate", "after_gate"}
    assert layer.screening_trace["score"].shape == torch.Size([1, 2, 4, 4])
    assert layer.screening_trace["score"].device.type == "cpu"
    assert not layer.screening_trace["score"].requires_grad
    assert torch.isfinite(layer.screening_trace["score"]).all()
    assert layer.screening_trace["before_gate"].shape == torch.Size([1, 2, 4, 4])
    assert not layer.screening_trace["before_gate"].requires_grad
    assert torch.isfinite(layer.screening_trace["before_gate"]).all()
    assert layer.screening_trace["after_gate"].shape == torch.Size([1, 2, 4, 4])
    assert not layer.screening_trace["after_gate"].requires_grad
    assert torch.isfinite(layer.screening_trace["after_gate"]).all()

    layer.set_trace_screening(False)
    _output = layer(hidden_states, position_ids=position_ids)

    assert layer.screening_trace == {}


def test_multiscreen_initial_scalar_parameters_match_paper_values():
    model = MultiScreenForCausalLM(
        hidden_dim=64,
        num_heads=4,
        num_blocks=2,
        vocab_size=16,
        window_threshold=10.0,
    )

    layer = model.layers[0]

    torch.testing.assert_close(model.token_embedding.scale, torch.tensor([1.0]))
    torch.testing.assert_close(model.head.scale, torch.tensor([8.0]))
    torch.testing.assert_close(
        layer._window_exponent,
        torch.linspace(0, math.log(10.0), 4),
    )
    torch.testing.assert_close(layer.acceptance, torch.full((4,), 0.5))
    torch.testing.assert_close(layer.scale, torch.tensor([1 / math.sqrt(4 * 2)]))


def test_init_multiscreen_wo_grad():
    num_blocks = 2
    hidden_dim = 64
    num_heads = 4
    window_threshold = 10.0

    tokenizer = ABCDTokenizer()
    vocab_size = tokenizer.vocab_size

    model = MultiScreenForCausalLM(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        window_threshold=window_threshold,
    )

    with torch.no_grad():
        _logits = model(
            input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            attention_mask=torch.tensor(
                [[1, 1, 0], [1, 1, 1]], dtype=torch.float
            ),  # Example mask
        )


def test_init_multiscreen_with_grad():
    num_blocks = 2
    hidden_dim = 64
    num_heads = 4
    window_threshold = 10.0

    tokenizer = ABCDTokenizer()
    vocab_size = tokenizer.vocab_size

    model = MultiScreenForCausalLM(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        window_threshold=window_threshold,
    )

    logits = model(
        input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        attention_mask=torch.tensor(
            [[1, 1, 0], [1, 1, 1]], dtype=torch.float
        ),  # Example mask
    )

    loss = logits.sum()
    loss.backward()


def test_multiscreen_gradient_checkpointing_with_grad():
    tokenizer = ABCDTokenizer()
    model = MultiScreenForCausalLM(
        hidden_dim=64,
        num_heads=4,
        num_blocks=2,
        vocab_size=tokenizer.vocab_size,
        window_threshold=10.0,
    )
    model.set_gradient_checkpointing(True)

    logits = model(
        input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float),
    )

    loss = logits.sum()
    loss.backward()
