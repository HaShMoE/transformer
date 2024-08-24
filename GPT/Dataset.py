import numpy as np
import itertools
import random
import locale
locale.getpreferredencoding = lambda: "UTF-8"

class Dataset:
    def __init__(self, path: str, batch_size: int, sequence_length: int):
        self._batch_size = batch_size

        with open(path, "r") as f:
            corpus = f.read()

        # Tokenize by splitting the text into characters
        words = corpus.split()
        chars = []
        for word in words:
            for ch in word:
                chars.append(ch)
            chars.append(" ")

        self.vocab_size = len(set(chars)) # Number of unique characters

        # Create a mapping from words to unique IDs
        self.char_to_id = {ch: i for i, ch in enumerate(set(chars))}

        # Store the inverse mapping from IDs to words
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}

        # Convert the words in the corpus to their corresponding IDs
        corpus = np.array([self.char_to_id[ch] for ch in chars]).astype(np.int32)

        crop_len = sequence_length + 1
        num_batches, ragged = divmod(corpus.size, batch_size * crop_len)
        if ragged:
            corpus = corpus[:-ragged]
        corpus = corpus.reshape([-1, crop_len])

        if num_batches < 10:
            raise ValueError(
                f"Only {num_batches} batches; consider a shorter "
                "sequence or a smaller batch."
            )

        self._ds = Dataset._infinite_shuffle(
            corpus, batch_size * 10
        )

    def __iter__(self):
        return self

    def __next__(self):
        """Yield next mini-batch."""
        batch = [next(self._ds) for _ in range(self._batch_size)]
        batch = np.stack(batch)
        # Create the language modeling observation/target pairs.
        return dict(
            input=batch[:, :-1], target=batch[:, 1:]
        )

    def ids_to_chars(self, ids):
        """Convert a sequence of char IDs to chars."""
        return [self.id_to_char[id] for id in ids]

    @staticmethod
    def _infinite_shuffle(iterable, buffer_size):
        """Infinitely repeat and shuffle data from iterable."""
        ds = itertools.cycle(iterable)
        buf = [next(ds) for _ in range(buffer_size)]
        random.shuffle(buf)
        while True:
            item = next(ds)
            idx = random.randint(0, buffer_size - 1)  # Inclusive.
            result, buf[idx] = buf[idx], item
            yield result
