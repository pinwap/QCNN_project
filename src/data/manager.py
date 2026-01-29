import logging
from typing import List, Optional, Tuple

from .dataset.factory import resolve_dataset
from .preprocessor.factory import resolve_preprocessors

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages the lifecycle of data loading and preprocessing.
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: str = "./data",
        n_train: int = 400,
        n_test: int = 100,
        preprocessors: Optional[List[str]] = None,
        target_dim: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = resolve_dataset(
            dataset_name, data_dir=data_path, n_train=n_train, n_test=n_test, **kwargs
        )

        if preprocessors:
            self.preprocessors = resolve_preprocessors(preprocessors)
        else:
            self.preprocessors = []

        if target_dim:
            from .preprocessor.ensure_feature_dimension import EnsureFeatureDimension

            self.preprocessors.append(EnsureFeatureDimension(target_dim))

    def get_data(self, as_numpy: bool = False) -> Tuple:
        """
        Loads, preprocesses, and splits data.
        """
        data, labels = self.dataset.load()

        for p in self.preprocessors:
            data = p(data)

        x_train, x_test, y_train, y_test = self.dataset.split(data, labels)

        if as_numpy:
            return x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()

        return x_train, y_train, x_test, y_test
