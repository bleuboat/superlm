import os
import torch

__all__ = ['get_device', 'set_device', 'set_seed', 'WorkSpace']

class _Device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device() -> torch.device:
    return _Device.device

def set_device(device: str | torch.device | int) -> None:
    _Device.device = torch.device(device)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

class WorkSpace:
    def __init__(self, folder: str, *, version: int = 1) -> None:
        self.folder = folder
        self.path = f'workspace/{folder}'
        self.model_path = f'{self.path}/model-{version}.pth'
        self.fig_path = f'{self.path}/model-{version}-loss.png'
        self._data = None

    @property
    def data(self) -> str | list[str]:
        if self._data is None:
            try:
                self._data = open(f'{self.path}/input.txt', 'r', encoding='utf-8').read()
            except FileNotFoundError:
                try:
                    self._data = []
                    files = os.listdir(f'{self.path}/inputs')
                    for file in files:
                        self._data.append(open(f'{self.path}/inputs/{file}', 'r', encoding='utf-8').read())
                except FileNotFoundError:
                    raise FileNotFoundError(f'Input not found. (at {self.path})')
        return self._data