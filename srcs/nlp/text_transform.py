import torch
import json
import os


from abc import ABC, abstractmethod
from collections import Counter
from srcs.nlp.tokenizer import PhonemeTokenizer, Tokenizer, WordTokenizer


def load_vocab(path):
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def save_vocab(path, vocab):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(vocab) + "\n")


def build_word_vocab(train_dataset, min_word_frequency=1):
    if min_word_frequency < 1:
        raise ValueError("min_word_frequency must be at least 1.")

    tokenizer = WordTokenizer()
    freqs = Counter()

    for label in train_dataset["label"]:
        freqs.update(tokenizer.tokenize(label))

    words = [
        word for word, frequency in freqs.items() if frequency >= min_word_frequency
    ]

    return sorted(words), freqs


class TextTransform(ABC):
    ignore_id = -1
    blank_token = Tokenizer.blank_token
    unk_token = Tokenizer.unk_token

    metric_names: tuple = ()

    @staticmethod
    @abstractmethod
    def vocab_paths(dir_path):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode_for_metrics(self, ids):
        pass

    @abstractmethod
    def reference_for_metrics(self, text):
        pass

    @classmethod
    def build_vocab(cls, tokens):
        return [*cls.special_tokens, *sorted(tokens)]

    @staticmethod
    def to_list(ids):
        if torch.is_tensor(ids):
            return ids.detach().cpu().tolist()

        return ids


class PhonemeTransform(TextTransform):
    special_tokens = (TextTransform.unk_token,)
    unk_id = 0

    comp_names = ("initial", "rhyme", "tone")
    comp_metric_names = ("per_i", "per_r", "per_t")
    metric_names = ("wer", *comp_metric_names)

    @staticmethod
    def vocab_paths(dir_path):
        return {
            "initial_path": os.path.join(dir_path, "initial.txt"),
            "rhyme_path": os.path.join(dir_path, "rhyme.txt"),
            "tone_path": os.path.join(dir_path, "tone.txt"),
            "lookup_path": os.path.join(dir_path, "phoneme_lookup.json"),
        }

    def __init__(
        self,
        initial_path,
        rhyme_path,
        tone_path,
        lookup_path,
        train_dataset=None,
        min_word_frequency=1,
    ):
        self.tokenizer = PhonemeTokenizer(lookup_path)
        self.paths = {
            "initial": initial_path,
            "rhyme": rhyme_path,
            "tone": tone_path,
        }

        if train_dataset is not None:
            word_vocab, freqs = build_word_vocab(
                train_dataset,
                min_word_frequency,
            )
            self.__build_vocabs(word_vocab, freqs, lookup_path)

        with open(lookup_path, encoding="utf-8") as file:
            lookup = json.load(file)

        self.known_words = {
            word for candidates in lookup.values() for word in candidates
        }
        self.tokenizer.lookup = lookup

        self.token2id = {}
        self.id2token = {}
        self.vocab_size = {}

        for name, path in self.paths.items():
            vocab = load_vocab(path)
            self.token2id[name] = {token: index for index, token in enumerate(vocab)}
            self.id2token[name] = dict(enumerate(vocab))
            self.vocab_size[name] = len(vocab)

    def __build_vocabs(self, word_vocab, freqs, lookup_path):
        comps = {name: set() for name in self.comp_names}
        lookup = {}

        for word in word_vocab:
            analysis = self.tokenizer.analyze(word)

            if not analysis["is_valid"]:
                continue

            phonemes = self.__to_phonemes(analysis)

            for name, phoneme in zip(self.comp_names, phonemes):
                comps[name].add(phoneme)

            key = self.tokenizer.merge_phoneme(phonemes)
            lookup.setdefault(key, []).append(word)

        for candidates in lookup.values():
            candidates.sort(key=lambda word: -freqs[word])

        for name, path in self.paths.items():
            save_vocab(path, self.build_vocab(comps[name]))

        with open(lookup_path, "w", encoding="utf-8") as file:
            json.dump(lookup, file, ensure_ascii=False, indent=2)

        self.tokenizer.lookup = lookup

    def __to_phonemes(self, analysis):
        return [
            analysis["initial"],
            self.tokenizer.merge_phoneme(
                [analysis["glide"], analysis["vowel"], analysis["final"]]
            ),
            analysis["tone"],
        ]

    def encode(self, text):
        labels = []

        for word in self.tokenizer.to_word(text):
            if word not in self.known_words:
                labels.append([self.unk_id] * len(self.comp_names))
                continue

            analysis = self.tokenizer.analyze(word)
            phonemes = self.__to_phonemes(analysis)

            if any(
                phoneme not in self.token2id[name]
                for name, phoneme in zip(self.comp_names, phonemes)
            ):
                labels.append([self.unk_id] * len(self.comp_names))
                continue

            labels.append(
                [
                    self.token2id[name][phoneme]
                    for name, phoneme in zip(self.comp_names, phonemes)
                ]
            )

        if not labels:
            return torch.empty((0, len(self.comp_names)), dtype=torch.long)

        return torch.tensor(labels, dtype=torch.long)

    def decode_for_metrics(self, ids):
        syllables = [
            [
                self.id2token[name].get(int(index), self.unk_token)
                for name, index in zip(self.comp_names, syllable_ids)
            ]
            for syllable_ids in self.to_list(ids)
            if not all(int(index) == self.ignore_id for index in syllable_ids)
        ]

        return self.__metrics(syllables, self.tokenizer.detokenize(syllables))

    def reference_for_metrics(self, text):
        sentence = " ".join(self.tokenizer.to_word(text))

        return self.__metrics(self.tokenizer.tokenize(text), sentence)

    def __metrics(self, syllables, sentence):
        results = {
            name: " ".join(syllable[index] for syllable in syllables)
            for index, name in enumerate(self.comp_metric_names)
        }
        results["wer"] = sentence

        return results


class WordTransform(TextTransform):
    special_tokens = Tokenizer.special_tokens

    blank_id = special_tokens.index(TextTransform.blank_token)
    unk_id = special_tokens.index(TextTransform.unk_token)

    metric_names = ("wer",)

    @staticmethod
    def vocab_paths(dir_path):
        return {"word_path": os.path.join(dir_path, "word.txt")}

    def __init__(self, word_path, train_dataset=None, min_word_frequency=1):
        self.tokenizer = WordTokenizer()
        self.path = word_path

        if train_dataset is not None:
            words, _ = build_word_vocab(train_dataset, min_word_frequency)
            save_vocab(word_path, self.build_vocab(words))

        vocab = load_vocab(word_path)
        self.token2id = {token: index for index, token in enumerate(vocab)}
        self.id2token = dict(enumerate(vocab))
        self.vocab_size = len(vocab)

    def encode(self, text):
        ids = [
            self.token2id.get(word, self.unk_id)
            for word in self.tokenizer.tokenize(text)
        ]

        if not ids:
            return torch.empty(0, dtype=torch.long)

        return torch.tensor(ids, dtype=torch.long)

    def decode_for_metrics(self, ids):
        words = [
            self.id2token.get(int(index), self.unk_token)
            for index in self.to_list(ids)
            if int(index) != self.ignore_id
        ]

        return self.__metrics(words)

    def reference_for_metrics(self, text):
        return self.__metrics(self.tokenizer.tokenize(text))

    def __metrics(self, words):
        sentence = self.tokenizer.detokenize(words)

        return {"wer": sentence}
