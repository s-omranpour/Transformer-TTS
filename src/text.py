from phonemizer import phonemize
from num2fawords import ordinal_words

class TextProcessor:
    def __init__(self, vocab):
        self.special_tokens = ['<BOS>', '<EOS>', '<PAD>']
        self.vocab = vocab + self.special_tokens
        self.n_vocab = len(self.vocab)
        self.bos = '<BOS>'
        self.bos_id = self.vocab.index(self.bos)
        self.eos = '<EOS>'
        self.eos_id = self.vocab.index(self.eos)
        self.pad = '<PAD>'
        self.pad_id = self.vocab.index(self.pad)

    def encode(self, s):
        res = []
        for c in s:
            res += [self.vocab.index(c)]
        return [self.bos_id] + res + [self.eos_id]
    
    def decode(self, ids):
        res = ''
        for idx in ids:
            c = self.vocab[idx]
            if c not in self.special_tokens:
                res += c
        return res
                

    def __call__(self, text, to_phones=True, to_indices=False):
        words = text.strip().split()
        for i, word in enumerate(words):
            try:
                words[i] = ordinal_words(word)[:-1]
            except:
                pass
        res = ' '.join(words)
        if to_phones:
            res = phonemize(res, language='fa', backend='espeak', strip=True)
        if to_indices:
            return self.encode(res)
        return res
