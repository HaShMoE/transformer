import matplotlib.pyplot as plt
import umap
import seaborn as sns
import gensim
import numpy as np
import jax
import jax.numpy as jnp
import colorama

from nltk.data import find


class Utils:
    def plot_position_encodings(P, max_tokens, d_model):
        """Function that takes in a position encoding matrix and plots it."""

        plt.figure(figsize=(20, np.min([8, max_tokens])))
        im = plt.imshow(P, aspect="auto", cmap="Blues_r")
        plt.colorbar(im, cmap="blue")

        if d_model <= 64:
            plt.xticks(range(d_model))
        if max_tokens <= 32:
            plt.yticks(range(max_tokens))
        plt.xlabel("Embedding index")
        plt.ylabel("Position index")
        plt.show()


    def plot_image_patches(patches):
        """Function that takes in a list of patches and plots them."""
        axes = []
        fig = plt.figure(figsize=(25, 25))
        for a in range(patches.shape[1]):
            axes.append(fig.add_subplot(1, patches.shape[1], a + 1))
            plt.imshow(patches[0][a])
        fig.tight_layout()
        plt.show()


    def plot_projected_embeddings(embeddings, labels):
        """Function that takes in a list of embeddings projects them onto a 2D space and plots them using UMAP."""

        projected_embeddings = umap.UMAP().fit_transform(embeddings)

        plt.figure(figsize=(15, 8))
        plt.title("Projected text embeddings")
        sns.scatterplot(
            x=projected_embeddings[:, 0], y=projected_embeddings[:, 1], hue=labels
        )
        plt.show()


    def plot_attention_weight_matrix(weight_matrix, x_ticks, y_ticks):
        """Function that takes in a weight matrix and plots it with custom axis ticks"""
        plt.figure(figsize=(15, 7))
        ax = sns.heatmap(weight_matrix, cmap="Blues")
        plt.xticks(np.arange(weight_matrix.shape[1]) + 0.5, x_ticks)
        plt.yticks(np.arange(weight_matrix.shape[0]) + 0.5, y_ticks)
        plt.title("Attention matrix")
        plt.xlabel("Attention score")
        plt.show()


    def get_word2vec_embedding(words):
        """
        Function that takes in a list of words and returns a list of their embeddings,
        based on a pretrained word2vec encoder.
        """
        word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))
        model = gensim.models.KeyedVectors.load_word2vec_format(
            word2vec_sample, binary=False
        )

        output = []
        words_pass = []
        for word in words:
            try:
                output.append(jnp.array(model.get_vector(word)))
                words_pass.append(word)
            except:
                pass

        embeddings = jnp.array(output)
        del model  # free up space again
        return embeddings, words_pass


    def remove_punctuation(text):
        """Function that takes in a string and removes all punctuation."""
        import re

        text = re.sub(r"[^\w\s]", "", text)
        return text

    def print_sample(prompt: str, sample: str):
        print(colorama.Fore.MAGENTA + prompt, end="")
        print(colorama.Fore.BLUE + sample)
        print(colorama.Fore.RESET)