from src.superlm_fork import *
space = WorkSpace('css', seed=42)
space.config(special_tokens=('bos', 'eos', 'pad'))
space.config(n_layer=4, n_embd=256, n_head=4, dropout=0.2)
space.config(epochs=10, block_size=1024, batch_size=2, accumulation_steps=1)
space.config(lr=3e-4)
# space.train()
print(space.generate('test'))