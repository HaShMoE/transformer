import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools
import livelossplot
from livelossplot import PlotLosses

from Dataset import Dataset
from model import LM

class Trainer:
    def __init__(self, input_text, validation_text, d_model, num_heads, num_layers, widening_factor, LR, batch_size, seq_length, dropout):
        self.input_text = input_text
        self.validation_text = validation_text
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.LR = LR
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dropout = dropout

        self.train_dataset = Dataset(input_text, batch_size, seq_length)
        self.validation_dataset = Dataset(validation_text, batch_size, seq_length)
        self.vocab_size = self.train_dataset.vocab_size

        self.rng = jax.random.PRNGKey(42)

        self.lm = LM(num_heads=num_heads, num_layers=num_layers, d_m=d_model, vocab_size=self.vocab_size, widening_factor=widening_factor, dropout=dropout)
        batch = next(self.train_dataset)
        mask = jnp.tril(np.ones((batch['input'].shape[1], batch['input'].shape[1])))
        self.params = self.lm.init(self.rng, batch['input'], mask)

        self.optimizer = optax.adam(LR, b1=0.9, b2=0.99)
        self.optimizer_state = self.optimizer.init(self.params)

    @functools.partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, optimizer_state, batch):
        def sequence_loss_fn(logits, targets):
            vocab_size = self.vocab_size
            target_labels = jax.nn.one_hot(targets, vocab_size)
            assert logits.shape == target_labels.shape
            mask = jnp.greater(targets, 0)
            loss = -jnp.sum(target_labels * jax.nn.log_softmax(logits), axis=-1)
            sequence_loss = jnp.sum(loss * mask) / jnp.sum(mask)
            bpc = sequence_loss / jnp.log(2)  # Convert loss to bpc
            return bpc
        
        def loss_fn(params):
            T = batch['input'].shape[1]
            logits = self.lm.apply(params, batch['input'], jnp.tril(np.ones((T, T))))
            loss = sequence_loss_fn(logits, batch['target'])
            return loss

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, optimizer_state = self.optimizer.update(gradients, optimizer_state)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss

    @functools.partial(jax.jit, static_argnums=(0,))
    def validation_step(self, params, batch):
        def sequence_loss_fn(logits, targets):
            vocab_size = self.vocab_size
            target_labels = jax.nn.one_hot(targets, vocab_size)
            assert logits.shape == target_labels.shape
            mask = jnp.greater(targets, 0)
            loss = -jnp.sum(target_labels * jax.nn.log_softmax(logits), axis=-1)
            sequence_loss = jnp.sum(loss * mask) / jnp.sum(mask)
            bpc = sequence_loss / jnp.log(2)  # Convert loss to bpc
            return bpc
        
        T = batch['input'].shape[1]
        logits = self.lm.apply(params, batch['input'], jnp.tril(jnp.ones((T, T))))
        loss = sequence_loss_fn(logits, batch['target'])
        return loss

    def train(self, max_steps, log_every, valid_every):
        plotlosses = PlotLosses()
        losses = []
        val_losses = []

        for step in range(max_steps):
            batch = next(self.train_dataset)
            self.params, self.optimizer_state, loss = self.train_step(self.params, self.optimizer_state, batch)
            losses.append(loss)

            if step % log_every == 0:
                loss_ = jnp.array(losses).mean()
                plotlosses.update({"train_bpc": loss_})  # Log bpc instead of generic loss
                losses = []

            if step % valid_every == 0:
                val_losses = []
                for _ in range(5):  # Compute validation bpc over 5 batches
                    batch_val = next(self.validation_dataset)
                    val_loss = self.validation_step(self.params, batch_val)
                    val_losses.append(val_loss)
                
                avg_val_loss = jnp.array(val_losses).mean()
                plotlosses.update({"val_bpc": avg_val_loss})  # Log bpc for validation
                
                plotlosses.send()

        return self.params

