from superlm import *
space = WorkSpace("luxun/test", seed=42, dtype=dtypes.bfloat16)
space.config(model_type="transformer", n_embd=256)
space.config(epochs=1, batch_size=4, accumulation_steps=1)
space.info()
space.train(length=200)
