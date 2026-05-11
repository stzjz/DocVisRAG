from abc import ABC, abstractmethod
from typing import Dict, List


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict]:
        raise NotImplementedError
