import math
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

class SequenceToQKV(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, X):
        initializer = nn.initializers.variance_scaling(scale=0.5, mode="fan_in", distribution="truncated_normal")

        # this can also be one layer, how do you think you would do it?
        q_layer = nn.Dense(self.output_size, kernel_init=initializer)
        k_layer = nn.Dense(self.output_size, kernel_init=initializer)
        v_layer = nn.Dense(self.output_size, kernel_init=initializer)

        Q = q_layer(X)
        K = k_layer(X)
        V = v_layer(X)

        return Q, K, V

class MultiHeadAttention(nn.Module):
    num_heads: int
    d_m: int
    dropout: float = 0.0

    def setup(self):
        self.sequence_to_qkv = SequenceToQKV(self.d_m)
        initializer = nn.initializers.variance_scaling(
            scale=0.5, mode="fan_in", distribution="truncated_normal")
        self.Wo = nn.Dense(self.d_m, kernel_init=initializer)
        self.dr_func = nn.Dropout(self.dropout, deterministic=(self.dropout>0))

    def scaled_dot_product_attention(query, key, value , dropout: nn.Dropout, mask=None):
        d_k = key.shape[-1]
        T_k = key.shape[-2]
        T_q = query.shape[-2]
        logits = jnp.matmul(query, jnp.swapaxes(key, -2, -1))
        scaled_logits = logits / jnp.sqrt(d_k)

        if mask is not None:
            scaled_logits = jnp.where(mask[:T_q, :T_k], scaled_logits, -jnp.inf)

        scaled_logits = dropout(scaled_logits)
        attention_weights = jax.nn.softmax(scaled_logits, axis=-1)
        attention = jnp.matmul(attention_weights, value)
        return attention, attention_weights

    def __call__(self, X=None, Q=None, K=None, V=None, mask=None, return_weights=False):
        if None in [Q, K, V]:
            assert not X is None, "X has to be provided if either Q,K,V not provided"

        # project all data to Q, K, V
        Q, K, V = self.sequence_to_qkv(X)

        # get the batch size, sequence length and embedding size
        B, T, d_m = K.shape

        # calculate heads embedding size (d_m/N)
        head_size = d_m // self.num_heads

        # B,T,d_m -> B, T, N, dm//N -> B, N, T, dm//N
        q_heads = Q.reshape(B, -1, self.num_heads, head_size).swapaxes(1, 2)
        k_heads = K.reshape(B, -1, self.num_heads, head_size).swapaxes(1, 2)
        v_heads = V.reshape(B, -1, self.num_heads, head_size).swapaxes(1, 2)

        attention, attention_weights = MultiHeadAttention.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, self.dr_func, mask
        )

        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, d_m) - re-assemble all head outputs
        attention = attention.swapaxes(1, 2).reshape(B, -1, d_m)

        # apply Wo
        X_new = self.Wo(attention)

        if return_weights:
            return X_new, attention_weights
        else:
            return X_new
        
class FeedForwardBlock(nn.Module):
    """A 2-layer MLP which widens then narrows the input."""
    widening_factor: int = 4
    init_scale: float = 0.25
    dropout: float = 0.0


    @nn.compact
    def __call__(self, x):
        d_m = x.shape[-1]
        layer1_size = self.widening_factor * d_m

        initializer = nn.initializers.variance_scaling(
            scale=self.init_scale, mode='fan_in', distribution='truncated_normal',
        )
        layer1 = nn.Dense(layer1_size, kernel_init=initializer)
        layer2 = nn.Dense(d_m, kernel_init=initializer)
        dropout = nn.Dropout(self.dropout, deterministic=(self.dropout>0))

        x = jax.nn.gelu(layer1(x))
        x = dropout(x)
        x = layer2(x)
        return x
  
class AddNorm(nn.Module):
    """A block that impliments the add and norm block"""

    @nn.compact
    def __call__(self, x, processed_x):

        added = x + processed_x
        normalised = nn.LayerNorm(reduction_axes=-1, use_scale=True, use_bias=True)
        return normalised(added)

class DecoderBlock(nn.Module):
    """
    Transformer decoder block.

    Args:
        num_heads: The number of heads to be used in the MHA block.
        d_m: Token embedding size
        widening factor: The size of the hidden layer will be d_m * widening_factor.
    """

    num_heads: int
    d_m: int
    widening_factor: int = 4
    dropout: float = 0.0

    def setup(self):
        self.mha = MultiHeadAttention(self.num_heads, self.d_m, dropout=self.dropout)
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()
        self.MLP = FeedForwardBlock(widening_factor=self.widening_factor, dropout=self.dropout)

    def __call__(self, X, mask=None, return_att_weight=True):
        """
        Args:
        X: Batch of tokens being fed into the decoder, with shape [B, T_decoder, d_m]
        mask [optional, default=None]: Mask to be applied, with shape [T_decoder, T_decoder].
        return_att_weight [optional, default=True]: Whether to return the attention weights.
        """

        attention, attention_weights_1 = self.mha(X, mask=mask, return_weights=True)

        X = self.add_norm1(X, attention)

        projection = self.MLP(X)
        X = self.add_norm2(X, projection)

        return (X, attention_weights_1) if return_att_weight else X
    
class LM(nn.Module):
    """
    Transformer decoder consisting of several layers of decoder blocks.

    Args:
        num_heads: The number of heads to be used in the MHA block.
        num_layers: The number of decoder blocks to be used.
        d_m: Token embedding size
        vocab_size: The size of the vocabulary
        widening_factor: The size of the hidden layer will be d_m * widening_factor.
    """
    num_heads: int
    num_layers: int
    d_m: int
    vocab_size: int
    widening_factor: int = 4
    dropout: float = 0.0

    def setup(self):
        self.blocks = [
            DecoderBlock(self.num_heads, self.d_m, self.widening_factor, self.dropout)
            for _ in range(self.num_layers)
        ]
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_m) # convert tokens to embedding
        self.pred_layer = nn.Dense(self.vocab_size)

    def return_frequency_pe_matrix(token_sequence_length, token_embedding):
        assert token_embedding % 2 == 0, "token_embedding should be divisible by two"

        P = jnp.zeros((token_sequence_length, token_embedding))
        positions = jnp.arange(0, token_sequence_length)[:, jnp.newaxis]

        i = jnp.arange(0, token_embedding, 2)
        frequency_steps = jnp.exp(i * (-math.log(10000.0) / token_embedding))
        frequencies = positions * frequency_steps

        P = P.at[:, 0::2].set(jnp.sin(frequencies))
        P = P.at[:, 1::2].set(jnp.cos(frequencies))

        return P

    def __call__(self, X, mask=None, return_att_weights=False):
        """
        Args:
        X: Batch of tokens being fed into the decoder, with shape [B, T_decoder, d_m]
        mask [optional, default=None]: Mask to be applied, with shape [T_decoder, T_decoder].
        return_att_weight [optional, default=True]: Whether to return the attention weights.
        """

        # convert a token id to a d_m dimensional vector
        X = self.embedding(X)
        sequence_len = X.shape[-2]
        positions = LM.return_frequency_pe_matrix(sequence_len, self.d_m)
        X = X + positions

        if return_att_weights:
            att_weights = []
        block_n = 0
        for block in self.blocks:
            out = block(X, mask, return_att_weights)
            if return_att_weights:
                X = out[0]
                att_weights.append(out[1])
            else:
                X = out

        # apply a linear layer and softmax to calculate our logits over tokens
        logits = nn.log_softmax(self.pred_layer(X))

        return (
            logits if not return_att_weights else (logits, jnp.array(att_weights).swapaxes(0, 1))
        )
