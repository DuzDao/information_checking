from underthesea import word_tokenize

def get_vocab_size(dataset):
    vocab = {}

    for i in range(len(dataset)):
        claim_tokens = word_tokenize(dataset[i]["claim"])  
        for token in claim_tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1

        evidence_tokens = word_tokenize(dataset[i]["evidence"])
        for token in evidence_tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1

    vocab_size = len(vocab)
    return vocab_size