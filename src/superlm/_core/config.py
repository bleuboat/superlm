__all__ = ['ModelConfig', 'GenerationConfig']

class Config(dict):
    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __getattr__(self, name: str):
        return self[name]

    def copy(self):
        return type(self)(**self)

    def to_dict(self) -> dict:
        return dict(self)

class ModelConfig(Config):
    def __init__(self, **kwargs) -> None:
        self.vocab_size = kwargs.pop('vocab_size', None)
        self.n_layer = kwargs.pop('n_layer', 4)
        self.n_embd = kwargs.pop('n_embd', 256)
        self.n_inner = kwargs.pop('n_inner', self.n_embd * 4)
        self.n_head = kwargs.pop('n_head', self.n_embd // 64)
        self.rope_theta = kwargs.pop('rope_theta', 10000)
        self.dropout = kwargs.pop('dropout', 0.1)
        self.eps = kwargs.pop('eps', 1e-8)
        self.tie_word_embeddings = kwargs.pop('tie_word_embeddings', False)
        assert self.n_embd % self.n_head == 0

class GenerationConfig(Config):
    def __init__(self, **kwargs) -> None:
        self.max_length = kwargs.pop('max_length', 20)
        self.max_new_tokens = kwargs.pop('max_new_tokens', None)
        self.max_time = kwargs.pop('max_time', None)

        self.do_sample = kwargs.pop('do_sample', False)
        self.temperature = kwargs.pop('temperature', 1.0)
        self.top_k = kwargs.pop('top_k', 50)
        self.top_p = kwargs.pop('top_p', 1.0)
        self.repetition_penalty = kwargs.pop('repetition_penalty', 1.0)

        self.pad_token_ix = kwargs.pop('pad_token_ix', None)
        self.bos_token_ix = kwargs.pop('bos_token_ix', None)
        self.eos_token_ix = kwargs.pop('eos_token_ix', None)