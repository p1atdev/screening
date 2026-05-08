from screening.tokenizer import ABCDTokenizer
from screening.abcd_digits import generate_by_line_count, generate_by_token_count


def test_generate_by_line_count():
    n_lines = 100
    depth = 0.5
    n_trials = 10

    samples = generate_by_line_count(
        n_lines=n_lines,
        depth=depth,
        n_trials=n_trials,
    )
    assert len(samples) == n_trials
    for sample in samples:
        print(sample)
        assert sample.count("\n") == n_lines - 1


def test_generate_by_token_count():
    n_tokens = 512
    depth = 0.5
    n_trials = 10

    tokenizer = ABCDTokenizer()
    samples = generate_by_token_count(
        n_tokens=n_tokens,
        depth=depth,
        tokenizer=tokenizer,
        n_trials=n_trials,
    )
    assert len(samples) == n_trials
    for sample in samples:
        print(sample)
        tokenized = tokenizer(sample)
        assert len(tokenized.input_ids) <= n_tokens
