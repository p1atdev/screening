# Copied from https://github.com/ken-nakanishi/abcdigits

# MIT License

# Copyright (c) 2026 Ken Nakanishi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import string


def generate_by_line_count(n_lines, depth, n_digits=6, n_trials=1000):
    assert n_lines >= 50, f"n_line must be at least 50"
    rng = random.Random(123)
    out = []
    for _ in range(n_trials):
        digits = rng.sample(range(10 ** (n_digits - 1), 10**n_digits), 26)
        txts = [f"{s}={d}\n" for s, d in zip(string.ascii_uppercase, digits)]
        weights = [0] + [2**i for i in range(25)]
        rng.shuffle(weights)
        needle_idx = weights.index(0)

        idxs = list(range(26))
        idxs.pop(needle_idx)
        idxs += rng.choices(range(26), weights=weights, k=n_lines - 27)
        rng.shuffle(idxs)
        idxs.insert(int(len(idxs) * depth), needle_idx)
        idxs.append(needle_idx)
        out.append("".join([txts[idx] for idx in idxs]).rstrip())
    return out


def generate_by_token_count(n_tokens, depth, tokenizer, n_digits=6, n_trials=1000):
    rng = random.Random(123)
    out = []
    for _ in range(n_trials):
        digits = rng.sample(range(10 ** (n_digits - 1), 10**n_digits), 26)
        txts = [f"{s}={d}\n" for s, d in zip(string.ascii_uppercase, digits)]
        lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in txts]
        weights = [0] + [2**i for i in range(25)]
        rng.shuffle(weights)
        needle_idx = weights.index(0)

        tok_len = sum(lengths) + lengths[needle_idx]
        idxs = list(range(26))
        idxs.pop(needle_idx)
        assert tok_len * 2 < n_tokens, f"n_tokens must be at least {tok_len}*2."
        while tok_len < n_tokens:
            idx = rng.choices(range(26), weights=weights)[0]
            idxs.append(idx)
            tok_len += lengths[idx]
        rng.shuffle(idxs)
        idxs.insert(int(len(idxs) * depth), needle_idx)
        idxs.append(needle_idx)
        out.append("".join([txts[idx] for idx in idxs]).rstrip())
    return out
