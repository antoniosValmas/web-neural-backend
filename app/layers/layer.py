
from abc import ABC, abstractmethod
from typing import Any, Tuple


class Layer(ABC):
    @abstractmethod
    def getKerasLayer(self) -> Tuple[Any, Any]:
        pass
