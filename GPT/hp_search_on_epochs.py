from itertools import product
import jax
from trainer_on_epochs import Trainer
print("hello", flush=True)
jax.debug.print("hello")
jax.debug.print("outside loop")
jax.debug.print("inside loop")
# Define the hyperparameter space
hyperparameter_space = {
    'd_model': [128, 256, 512],
    'num_heads': [2, 4, 8],
    'num_layers': [1, 2, 3],
    'widening_factor': [2, 4, 8],
    'LR': [1e-3, 2e-3, 5e-3],
    'dropout': [0.1, 0.2, 0.3],
}

best_val_loss = float('inf')
best_hyperparams = None
best_params = None
print("outside loop")
# Grid search over the hyperparameter space
for d_model, num_heads, num_layers, widening_factor, LR, dropout in product(
        hyperparameter_space['d_model'],
        hyperparameter_space['num_heads'],
        hyperparameter_space['num_layers'],
        hyperparameter_space['widening_factor'],
        hyperparameter_space['LR'],
        hyperparameter_space['dropout']
    ):
    print("inside loop")
    print(f"Training with d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, widening_factor={widening_factor}, LR={LR}, dropout={dropout}")
    
    trainer = Trainer(
        input_text="../LMDatasets/nchlt_text.xh.train",
        validation_text="../LMDatasets/nchlt_text.xh.train",
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        widening_factor=widening_factor,
        LR=LR,
        batch_size=32,
        seq_length=64,
        dropout=dropout
    )
    
    final_params = trainer.train(max_steps=10, log_every=2000, valid_every=2000)
    
    # Evaluate validation loss after training
    val_loss = trainer.validation_step(trainer.params, next(trainer.validation_dataset))
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = {
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'widening_factor': widening_factor,
            'LR': LR,
            'dropout': dropout
        }
        best_params = final_params

print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best validation loss: {best_val_loss}")

# Optionally save the best model parameters
import flax.serialization

with open('best_model_params.pkl', 'wb') as f:
    f.write(flax.serialization.to_bytes(best_params))
