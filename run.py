from superlm import WorkSpace, dtypes
space = WorkSpace('backrooms2', seed=42, dtype=dtypes.bfloat16)
space.config(model_type='transformer')
space.config(epochs=1, batch_size=2, accumulation_steps=1)
space.info()
space.train(length=200)
# space.generate('Level 0', stream=True)