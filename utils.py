import dataclasses


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


@dataclasses.dataclass
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    hidden_size: int
    max_model_len: int

    @property
    def head_dim(self):
        return self.hidden_size // self.num_qo_heads


LLAMA2_7B = ModelConfig(
    num_layers=32,
    num_qo_heads=32,
    num_kv_heads=32,
    hidden_size=4096,
    max_model_len=4096,
)

LLAMA3_1_8B = ModelConfig(
    num_layers=32,
    num_qo_heads=32,
    num_kv_heads=8,
    hidden_size=4096,
    max_model_len=131072,
)

models = {
    "meta-llama/Llama-2-7b-hf": LLAMA2_7B,
    "meta-llama/Meta-Llama-3.1-8B": LLAMA3_1_8B,
}
