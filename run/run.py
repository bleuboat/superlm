from superlm import *
space = WorkSpace("luxun/test", seed=42, dtype=dtypes.bfloat16)
space.config(
    model_type="transformer",
    n_layer=6,
    n_embd=768,
    gradient_checkpointing=True,
)
space.config(epochs=1, batch_size=1, accumulation_steps=1)
space.info()
space.train(length=200)
