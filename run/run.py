from superlm import *
space = WorkSpace('luxun/test', seed=42, dtype=dtypes.bfloat16)
space.config(model_type='transformer', n_embd=64)
space.config(epochs=1, batch_size=2, accumulation_steps=1)
space.info()
space.train(length=200)