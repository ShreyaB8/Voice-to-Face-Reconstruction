from typing import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    model: nn.Sequential

    def __init__(
        self, input_size: int, output_size: int,
        hidden_size: int | Sequence[int],
        hidden_layer_num: int,
        dropout_probability: float | Sequence[float],
    ):
        super().__init__()

        if isinstance(hidden_size, int):
            hidden_size = tuple(hidden_size for _ in range(hidden_layer_num))
        assert isinstance(hidden_size, Sequence)
        if len(hidden_size) == 1:
            hidden_size = [hidden_size[0] for _ in range(hidden_layer_num)]
        if len(hidden_size) != hidden_layer_num:
            raise ValueError(f'{len(hidden_size) = } and {hidden_layer_num = } does not match.')
        
        if isinstance(dropout_probability, float):
            dropout_probability = tuple(dropout_probability for _ in range(hidden_layer_num))
        assert isinstance(dropout_probability, Sequence)
        if len(dropout_probability) == 1:
            dropout_probability = [dropout_probability[0] for _ in range(hidden_layer_num)]
        if len(dropout_probability) != hidden_layer_num:
            raise ValueError(f'{len(dropout_probability) = } and {hidden_layer_num = } does not match.')

        models = [
            nn.Linear(input_size, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.GELU(),
            nn.Dropout1d(p=dropout_probability[0])
        ]
        for i in range(1, hidden_layer_num):
            models.extend([
                nn.Linear(hidden_size[i - 1], hidden_size[i]),
                nn.BatchNorm1d(hidden_size[i]),
                nn.GELU(),
                nn.Dropout1d(p=dropout_probability[i])
            ])
        models.extend([nn.Linear(hidden_size[-1], output_size)])
        
        for model in models:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight)
                nn.init.zeros_(model.bias)

        self.model = nn.Sequential(*models)
    
    def forward(self, x: Tensor):
        x = self.model(x)
        return x
