from superlm import *
workspace = WorkSpace('css', version=2)
model, tokenizer = Trainer.load(workspace, train=False)
streamer = Streamer(tokenizer)

while True:
    print('----')
    user_input = input('请输入开头：')
    try:
        print('AI生成：')
        model.generate(
            tokenizer([user_input], special_tokens=('bos')),
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            # repetition_penalty=1.1,
        )
    except KeyboardInterrupt:
        print('\033[31m[用户中断]\033[0m', end='')
        streamer.end()
    except TokenNotFoundError as e:
        print(f'\033[31m{e}\033[0m')