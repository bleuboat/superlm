from superlm import *
workspace = WorkSpace('backrooms', version=1)
model, tokenizer, trainer = Trainer.load(workspace)
trainer.block_size = 512
trainer.batch_size = 8
trainer.accumulation_steps = 4

for step in trainer:
    if step % 50 == 0:
        trainer.log()
        
    if step % 100 == 0:
        trainer.sample('标题：', 200)

    if step % 200 == 0:
        trainer.save()

trainer.show()