import torch
import numpy as np
from sklearn.model_selection import train_test_split


class TabPFNDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray = None,
        regression: bool = False,
        batch_size: int = 1024,
        max_features: int = 100,
    ):
        """
        :param: max_features: number of features the tab pfn model has been trained on
        """

        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test        
        self.y_test = y_test
        self.regression = regression

        if self.y_test is None:
            self.y_test = np.zeros((self.x_test.shape[0],)) - 1

        self.batch_size = batch_size
        self.max_features = max_features
        self.n_train_observations = x_train.shape[0]

        mean, std = self.calc_mean_std(x_train)
        self.x_train = self.normalize_by_mean_std(self.x_train, mean, std)
        self.x_test = self.normalize_by_mean_std(self.x_test, mean, std)

        self.x_train = self.normalize_by_feature_count(self.x_train, max_features)
        self.x_test = self.normalize_by_feature_count(self.x_test, max_features)

        self.x_train = self.extend_features(self.x_train, max_features=max_features)
        self.x_test = self.extend_features(self.x_test, max_features=max_features)

        self.x_tests = self.split_in_chunks(self.x_test, batch_size)
        self.y_tests = self.split_in_chunks(self.y_test, batch_size)

        # This single eval pos is necessary for the tab pfn forward pass
        # It's a marker for when the training data ends and the test data starts
        # We push the whole training data through the model, unless it's bigger than the batch size
        self.single_eval_pos = min(self.n_train_observations, self.batch_size)


    def __len__(self):
        return len(self.x_tests)

    def __getitem__(self, idx):

        train_size = self.single_eval_pos

        train_indices = np.random.choice(
            self.n_train_observations, 
            size=train_size, 
            replace=False
        )

        x_train = self.x_train[train_indices]
        y_train = self.y_train[train_indices]

        x_full = np.concatenate([x_train, self.x_tests[idx]], axis=0)
        x_full = torch.FloatTensor(x_full)

        if self.regression:
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(self.y_tests[idx])
        else:
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(self.y_tests[idx])

        return x_full, y_train, y_test
        

    def calc_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean and std of the training data
        """
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return mean, std
    

    def normalize_by_mean_std(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalizes the data by the mean and std
        """
        x = (x - mean) / std
        return x


    def normalize_by_feature_count(self, x: np.ndarray, max_features) -> np.ndarray:
        """
        An interesting way of normalization by the tabPFN paper
        """

        x = x * max_features / x.shape[1]
        return x



    def extend_features(self, x: np.ndarray, max_features) -> np.ndarray:
        """
        Increases the number of features to the number of features the tab pfn model has been trained on
        """
        x = np.concatenate([x, np.zeros((x.shape[0], max_features - x.shape[1]))], axis=1)
        return x


    def split_in_chunks(self, x: np.ndarray, batch_size: int) -> list[np.ndarray]:
        """
        Splits the data into chunks of size batch_size
        """

        n_chunks = int(np.ceil(x.shape[0] / batch_size))
        x_chunks = []

        for i in range(n_chunks):
            x_chunks.append(x[i * batch_size: (i + 1) * batch_size])

        return x_chunks
    





def TabPFNDatasetGenerator(
    x: np.ndarray, 
    y: np.ndarray, 
    regression: bool,
    batch_size: int = 1024,
    max_features: int = 100,
    split: float = 0.8,
):
    
    if regression:
        stratify = None
    else:
        stratify = y
        
    while True:

        x_train, x_test, y_train, y_test = train_test_split(
            x, 
            y, 
            train_size=split, 
            shuffle=True,
            stratify=stratify
        )

        static_dataset = TabPFNDataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            regression=regression,
            batch_size=batch_size,
            max_features=max_features
        )

        yield static_dataset