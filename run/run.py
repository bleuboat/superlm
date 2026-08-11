from superlm import *
space = WorkSpace('xiyouji', seed=42, dtype=dtypes.bfloat16)
space.config(model_type='transformer', n_embd=64)
space.config(epochs=2, batch_size=2, accumulation_steps=1)
space.info()
space.train(length=200)
# space.generate('Level 0', stream=True)