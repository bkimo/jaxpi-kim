import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
  #  config.mode = "eval"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Leray-alpha_unsteady_cylinder"
    wandb.name = "test1"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Leray-α parameter
    config.alpha = 0.002 # Need to do this automatially:Alph2Net

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "gelu"  # gelu works better than tanh
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 10.0, "embed_dim": 128}
    )
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200 
    training.num_time_windows = 2 #10

    training.inflow_batch_size = 128
    training.outflow_batch_size = 128
    training.noslip_batch_size = 128
    training.ic_batch_size = 128
    training.res_batch_size = 512

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = {
        "u_ic": 1.0,
        "v_ic": 1.0,
        "p_ic": 1.0,
        "u_in": 1.0,
        "v_in": 1.0,
        "u_out": 1.0,
        "v_out": 1.0,
        "u_noslip": 1.0,
        "v_noslip": 1.0,
        "ru": 1.0,
        "rv": 1.0,
        "rc": 1.0,
    }
    weighting.momentum = 0.9
    weighting.update_every_steps = 10  # 100 for grad norm and 1000 for ntk

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = True
    logging.log_grads = True
    logging.log_ntk = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
