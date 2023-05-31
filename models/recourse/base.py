from abc import ABC, abstractmethod

import numpy as np


class RecourseMethodBase(ABC):
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def provide_recourse(self, x: np.array):
        pass
