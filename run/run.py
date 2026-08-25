from superlm import *
space = WorkSpace("luxun/test", seed=42)
space.config(model_type="transformer", n_embd=64)
space.config(epochs=1, batch_size=2, accumulation_steps=1)
space.info()
space.train()
# space.load()
# space.generate(
#     "我",
#     stream=True,
#     top_k=None,
#     top_p=None,
#     top_h=0.9,
# )
