from dataclasses import dataclass
from typing import Optional, Self

import xarray as xr

from tabularbench.core.enums import (BenchmarkName, BenchmarkOrigin,
                                     DatasetSize, Task)
from tabularbench.data.datafile_name_maker import make_datafile_path


@dataclass
class Benchmark:
    name: BenchmarkName
    origin: BenchmarkOrigin
    task: Task
    dataset_size: Optional[DatasetSize]
    openml_dataset_ids: list[int]
    openml_dataset_names: list[str]

    @classmethod
    def create(
        cls, 
        name: BenchmarkName, 
        origin: BenchmarkOrigin,
        task: Task,  
        openml_dataset_ids: list[int],
        dataset_size: Optional[DatasetSize],
    ) -> Self:

        dataset_names = []
        for openml_dataset_id in openml_dataset_ids:
            datafile_path = make_datafile_path(origin, openml_dataset_id, dataset_size)
            dataset_name = xr.open_dataset(datafile_path).attrs['openml_dataset_name']
            dataset_names.append(dataset_name)

        return cls(
            name=name,
            origin=origin,
            task=task,
            dataset_size=dataset_size,
            openml_dataset_ids=openml_dataset_ids,
            openml_dataset_names=dataset_names
        )




BENCHMARKS = {
    BenchmarkName.DEBUG_CATEGORICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.DEBUG_CATEGORICAL_CLASSIFICATION,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44156, 44157, 44159, 45035, 45036, 45038, 45039]
    ),
    BenchmarkName.CATEGORICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_CLASSIFICATION,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44156, 44157, 44159, 45035, 45036, 45038, 45039]
    ),
    BenchmarkName.CATEGORICAL_REGRESSION: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_REGRESSION,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44055, 44056, 44059, 44061, 44062, 44063, 44065, 44066, 44068, 44069, 45041, 45042, 45043, 45045, 45046, 45047, 45048]
    ),
    BenchmarkName.NUMERICAL_REGRESSION: Benchmark.create(
        name=BenchmarkName.NUMERICAL_REGRESSION,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44132, 44133, 44134, 44136, 44137, 44138, 44139, 44140, 44141, 44142, 44143, 44144, 44145, 44146, 44147, 44148, 45032, 45033, 45034]
    ),
    BenchmarkName.NUMERICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.NUMERICAL_CLASSIFICATION,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44089, 44120, 44121, 44122, 44123, 44125, 44126, 44128, 44129, 44130, 45022, 45021, 45020, 45019, 45028, 45026]
    ),
    BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44156, 44157, 44159, 45035, 45036, 45038, 45039]
    ),
    BenchmarkName.CATEGORICAL_REGRESSION_LARGE: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_REGRESSION_LARGE,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44055, 44056, 44059, 44061, 44062, 44063, 44065, 44066, 44068, 44069, 45041, 45042, 45043, 45045, 45046, 45047, 45048]
    ),
    BenchmarkName.NUMERICAL_REGRESSION_LARGE: Benchmark.create(
        name=BenchmarkName.NUMERICAL_REGRESSION_LARGE,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44132, 44133, 44134, 44136, 44137, 44138, 44139, 44140, 44141, 44142, 44143, 44144, 44145, 44146, 44147, 44148, 45032, 45033, 45034]
    ),
    BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE: Benchmark.create(
        name=BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE,
        origin=BenchmarkOrigin.WHYTREES,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44089, 44120, 44121, 44122, 44123, 44125, 44126, 44128, 44129, 44130, 45022, 45021, 45020, 45019, 45028, 45026]
    ),
    BenchmarkName.TABZILLA_HARD: Benchmark.create(
        name=BenchmarkName.TABZILLA_HARD,
        origin=BenchmarkOrigin.TABZILLA,
        task=Task.CLASSIFICATION,
        dataset_size=None,
        openml_dataset_ids=[7, 10, 11, 14, 22, 25, 29, 31, 45, 50, 53, 219, 3561, 3711, 3797, 3896, 3917, 9890, 9910, 9952, 9956, 9957, 9977, 9981, 14964, 14969, 146065, 146606, 146607, 146818, 167119, 168335, 168337, 168911, 189354, 189356]
    ),
    BenchmarkName.TABZILLA_HARD_MAX_TEN_CLASSES: Benchmark.create(
        name=BenchmarkName.TABZILLA_HARD_MAX_TEN_CLASSES,
        origin=BenchmarkOrigin.TABZILLA,
        task=Task.CLASSIFICATION,
        dataset_size=None,
        openml_dataset_ids=[10, 11, 14, 22, 25, 29, 31, 45, 50, 53, 219, 3561, 3711, 3797, 3896, 3917, 9890, 9910, 9952, 9957, 9977, 9981, 14964, 14969, 146065, 146606, 146607, 146818, 167119, 168335, 168337, 168911, 189354, 189356]
    
    )
}


if __name__ == "__main__":
    print(BENCHMARKS)
    pass