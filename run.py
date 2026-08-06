from superlm import WorkSpace
space = WorkSpace('backrooms', seed=42)
space.config(model_type='bow')
space.config(n_layer=4, n_embd=64, n_head=4, block_size=1024, dropout=0.1)
space.config(epochs=1, batch_size=2, accumulation_steps=1)
space.info()
space.train(length=200)
space.generate('我', stream=True)