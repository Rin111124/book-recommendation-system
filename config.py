from __future__ import annotations

import pathlib
from dataclasses import dataclass, field


BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class DataConfig:
    raw_dir: pathlib.Path = DATA_DIR / "raw"
    processed_dir: pathlib.Path = DATA_DIR / "processed"
    models_dir: pathlib.Path = DATA_DIR / "models"
    books_file: pathlib.Path = field(init=False)
    ratings_file: pathlib.Path = field(init=False)
    users_file: pathlib.Path = field(init=False)

    def __post_init__(self) -> None:
        self.books_file = self.raw_dir / "Books.csv"
        self.ratings_file = self.raw_dir / "Ratings.csv"
        self.users_file = self.raw_dir / "Users.csv"


@dataclass
class ModelConfig:
    user_based_neighbors: int = 25
    item_based_neighbors: int = 20
    min_interactions_user: int = 5
    min_interactions_item: int = 5
    svd_latent_factors: int = 50
    svd_learning_rate: float = 0.005
    svd_reg: float = 0.02
    svd_epochs: int = 40


@dataclass
class WebConfig:
    default_user_id: int = 276729
    recommendations_limit: int = 10


data_config = DataConfig()
model_config = ModelConfig()
web_config = WebConfig()
