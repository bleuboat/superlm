from superlm import WorkSpace, dtypes
space = WorkSpace('backrooms', seed=42, dtype=dtypes.bfloat16)
space.del_checkpoint()
# space.config(model_type='bigram', n_embd=64)
# space.config(epochs=1, batch_size=1, accumulation_steps=1)
# space.info()
# space.train(length=200)
space.generate('我', stream=True, repetition_penalty=1.3)