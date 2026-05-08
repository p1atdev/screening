from screening.tokenizer import ABCDTokenizer


def test_abcd_tokenizer():
    tokenizer = ABCDTokenizer()

    text = "Hello, World!\nThis is a test."
    tokenized = tokenizer(text)
    print("Input text:", text)
    print("Token IDs:", tokenized.input_ids)
    print("Attention Mask:", tokenized.attention_mask)

    decoded_text = tokenizer.decode(tokenized.input_ids, skip_special_tokens=True)
    print("Decoded text:", decoded_text)
    assert text == decoded_text

    batch_texts = [text, "Another line.\nWith multiple lines.\nAnd punctuation!"]
    batch_tokenized = tokenizer.encode_batch(batch_texts)
    print("Batch Token IDs:", batch_tokenized.input_ids)
    print("Batch Attention Mask:", batch_tokenized.attention_mask)

    batch_decoded_texts = tokenizer.decode_batch(
        batch_tokenized.input_ids, skip_special_tokens=True
    )
    print("Batch Decoded Texts:", batch_decoded_texts)
    assert batch_texts == batch_decoded_texts
