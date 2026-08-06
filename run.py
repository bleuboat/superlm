from superlm import WorkSpace
space = WorkSpace('backrooms', seed=42)
space.config(model_type='transformer')
space.config(epochs=1, batch_size=2, accumulation_steps=1)
space.info()
space.train(length=200)
space.generate('我', stream=True)