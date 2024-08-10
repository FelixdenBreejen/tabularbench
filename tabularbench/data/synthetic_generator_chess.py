import numpy as np
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm


def synthetic_dataset_function_chess(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        max_grid_areas = 128
    ):

    n_classes = get_n_classes(max_classes)
    n_features = get_n_features(min_features, max_features)

    x = np.random.uniform(size=(n_samples, n_features))
    y = np.zeros(n_samples)

    for i in range(n_features):

        n_grid_areas = np.random.randint(1, max_grid_areas+1, size=1).item()
        grid_lines = np.random.uniform(0, 1, size=(n_grid_areas-1,))
        grid_lines.sort()
        grid_values = np.random.uniform(0, 1, size=(n_grid_areas,))
        x_indices = np.digitize(x[:, i], bins=grid_lines, right=True)
        x_i = grid_values[x_indices]
        y += x_i

    y = quantile_transform(y)
    y = put_in_buckets(y, n_classes)

    return x, y


def get_n_classes(max_classes: int) -> int:
    return np.random.randint(2, max_classes, size=1).item()
    
def get_n_features(min_features: int, max_features: int) -> int:
    if min_features == max_features:
        return min_features
    else:
        return np.random.randint(min_features, max_features, size=1).item()


def quantile_transform(z: np.ndarray) -> np.ndarray:
    quantile_transformer = QuantileTransformer(output_distribution='uniform')
    z = quantile_transformer.fit_transform(z.reshape(-1, 1)).flatten()
    return z


def put_in_buckets(z: np.ndarray, n_classes: int) -> np.ndarray:
    buckets = np.random.uniform(0, 1, size=(n_classes-1,))
    buckets.sort()
    buckets = np.hstack([buckets, 1])
    b = np.argmax(z <= buckets[:, None], axis=0)

    return b


def synthetic_dataset_generator_chess(**kwargs):

    while True:
        x, y = synthetic_dataset_function_chess(**kwargs)
        yield x, y



if __name__ == '__main__':

    generator = synthetic_dataset_generator_chess(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        max_grid_areas = 128
    )

    for _ in tqdm(range(1)):        
        x, y = next(generator)
        pass