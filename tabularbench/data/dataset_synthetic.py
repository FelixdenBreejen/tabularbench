from typing import Generator, Iterator

import torch
from tabularbench.data.preprocessor import Preprocessor

from tabularbench.models.tabPFN.synthetic_data import synthetic_dataset_generator
from tabularbench.sweeps.config_pretrain import ConfigPretrain



class SyntheticDataset(torch.utils.data.IterableDataset):

    def __init__(
        self, 
        cfg: ConfigPretrain,
        min_samples_support: int,
        max_samples_support: int,
        n_samples_query: int,
        min_features: int,
        max_features: int,
        max_classes: int,
        use_quantile_transformer: bool,
        use_feature_count_scaling: bool,
    ) -> None:
        
        self.cfg = cfg
        self.min_samples_support = min_samples_support
        self.max_samples_support = max_samples_support
        self.n_samples_query = n_samples_query
        self.n_samples = max_samples_support + n_samples_query
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling

    def __iter__(self) -> Iterator:

        self.synthetic_dataset_generator = synthetic_dataset_generator(
            n_samples=self.n_samples,
            min_features=self.min_features,
            max_features=self.max_features,
            max_classes=self.max_classes
        )

        return self.generator()
    

    def generator(self) -> Generator[dict[str, torch.Tensor], None, None]:
        
        while True:
            x, y = next(self.synthetic_dataset_generator)

            y = self.randomize_classes(y)
            x_support, y_support, x_query, y_query = self.split_into_support_and_query(x, y)

            preprocessor = Preprocessor(
                self.cfg.logger, 
                max_features=self.max_features,
                use_quantile_transformer=self.use_quantile_transformer,
                use_feature_count_scaling=self.use_feature_count_scaling,
            )

            x_support = preprocessor.fit_transform(x_support)
            x_query = preprocessor.transform(x_query)
            
            x_support = torch.tensor(x_support, dtype=torch.float32)
            x_query = torch.tensor(x_query, dtype=torch.float32)

            x_support, x_query = self.randomize_feature_order(x_support, x_query)

            yield {
                'x_support': x_support,
                'y_support': y_support,
                'x_query': x_query,
                'y_query': y_query
            }


    def randomize_classes(self, y):
            
        curr_classes = int(y.max().item()) + 1
        new_classes = torch.randperm(self.max_classes)
        mapping = { i: new_classes[i] for i in range(curr_classes) }
        y = torch.tensor([mapping[i.item()] for i in y], dtype=torch.long)

        return y


    def split_into_support_and_query(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        curr_samples = x.shape[0]

        n_samples_support = torch.randint(low=self.min_samples_support, high=self.max_samples_support, size=(1,)).item()

        rand_index = torch.randperm(curr_samples)
        rand_support_index = rand_index[:n_samples_support]
        rand_query_index = rand_index[n_samples_support:n_samples_support+self.n_samples_query]

        x_support = x[rand_support_index]
        y_support = y[rand_support_index]
        x_query = x[rand_query_index]
        y_query = y[rand_query_index]

        return x_support, y_support, x_query, y_query
    

    def randomize_feature_order(self, x_support: torch.Tensor, x_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        curr_features = x_support.shape[1]
        new_feature_order = torch.randperm(curr_features)

        x_support = x_support[:, new_feature_order]
        x_query = x_query[:, new_feature_order]

        return x_support, x_query
    
