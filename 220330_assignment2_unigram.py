class UnigramLanguageTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unknown_token = "<UNK>"
        


def main():
    vocab = ["hello", "world", "<UNK>"]
    tokenizer = UnigramLanguageTokenizer(vocab)

if __name__ == "__main__":
    main()