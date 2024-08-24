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

    @functools.partial(jax.jit, static_argnums=(3, 4))
    def train_step(params, optimizer_state, batch, apply_fn, update_fn):
        def sequence_loss_fn(logits, targets):
            """Compute the loss on data wrt params."""
            vocab_size = 108
            target_labels = jax.nn.one_hot(targets, vocab_size)
            assert logits.shape == target_labels.shape
            mask = jnp.greater(targets, 0)
            loss = -jnp.sum(target_labels * jax.nn.log_softmax(logits), axis=-1)
            sequence_loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return sequence_loss
        
        def loss_fn(params):
            T = batch['input'].shape[1]
            logits = apply_fn(params, batch['input'], jnp.tril(np.ones((T, T))))
            loss = sequence_loss_fn(logits, batch['target'])
            return loss
    @functools.partial(jax.jit, static_argnums=(3, 4))
    def validation_step(params, optimizer_state, batch, apply_fn, update_fn):
        def sequence_loss_fn(logits, targets):
            """Compute the loss on data wrt params."""
            vocab_size = 108
            target_labels = jax.nn.one_hot(targets, vocab_size)
            assert logits.shape == target_labels.shape
            mask = jnp.greater(targets, 0)
            loss = -jnp.sum(target_labels * jax.nn.log_softmax(logits), axis=-1)
            sequence_loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return sequence_loss
        
        def loss_fn(params):
            T = batch['input'].shape[1]
            logits = apply_fn(params, batch['input'], jnp.tril(np.ones((T, T))))
            loss = sequence_loss_fn(logits, batch['target'])
            return loss

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, optimizer_state = update_fn(gradients, optimizer_state)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss
    
    # all hyperparameters
    input_text = "LMDatasets/nchlt_text.xh.train"
    validation_text = "LMDatasets/nchlt_text.xh.valid"
    d_model = 128
    num_heads = 4
    num_layers = 1
    widening_factor = 2
    LR = 2e-3
    batch_size = 32
    seq_length = 64
    dropout = 0.2

    # set up the data
    train_dataset = Dataset(input_text, batch_size, seq_length)
    validation_dataset = Dataset(validation_text, batch_size, seq_length)
    vocab_size = train_dataset.vocab_size
    batch = next(train_dataset)

    rng = jax.random.PRNGKey(42) #Supposed to be key

    # initialise model
    lm = LM(num_heads=num_heads, num_layers=num_layers, d_m=d_model, vocab_size=vocab_size, widening_factor=widening_factor, dropout=dropout)
    mask = jnp.tril(np.ones((batch['input'].shape[1], batch['input'].shape[1])))
    params = lm.init(rng, batch['input'], mask)

    # set up the optimiser
    optimizer = optax.adam(LR, b1=0.9, b2=0.99)
    optimizer_state = optimizer.init(params)

    plotlosses = PlotLosses()

    MAX_STEPS = 10000
    LOG_EVERY = 300
    losses = []

    # Training loop
    print(vocab_size)
    for step in range(MAX_STEPS):
        batch = next(train_dataset)
        params, optimizer_state, loss = train_step(
            params, optimizer_state, batch, lm.apply, optimizer.update)
        losses.append(loss)
        if step % LOG_EVERY == 0:
            loss_ = jnp.array(losses).mean()
            plotlosses.update(
                {
                    "loss": loss_,
                }
            )
            plotlosses.send()
            losses = []


    #Predicting
    

    def generate_random(lm, params, id_2_char, char_2_id):
        '''
        Get the model output
        '''
        @functools.partial(jax.jit, static_argnums=(2, ))
        def generate_prediction(params, input, apply_fn):
            logits = apply_fn(params, input)
            argmax_out = jnp.argmax(logits, axis=-1)
            return argmax_out[0][-1].astype(int)

        prompt = "Hishaam ngu"
        print(prompt, end="")
        words = prompt.split()
        chars = []
        for word in words:
            for ch in word:
                chars.append(ch)
            chars.append(" ")

        # predict and append
        for i in range(40):
            input = jnp.array([[char_2_id[t] for t in chars]]).astype(int)
            prediction = generate_prediction(params, input, lm.apply)
            prediction = id_2_char[int(prediction)]
            chars.append(prediction)
            print(prediction, end="")

        return " ".join(chars)
    
    id_2_char = train_dataset.id_to_char
    char_2_id = train_dataset.char_to_id

    generated_text = generate_random(lm, params, id_2_char, char_2_id)