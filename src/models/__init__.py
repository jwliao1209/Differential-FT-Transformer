from typing import Union

from .ft_transformer import FTTransformer, FTTransformerClassifier, FTTransformerRegressor
from .diff_ft_transformer import DiffFTTransformer, DiffFTTransformerClassifier, DiffFTTransformerRegressor


def get_model(model_name: str) -> Union[FTTransformer, DiffFTTransformer]:
    match model_name:
        case 'fttc':
            return FTTransformerClassifier
        case 'fttr':
            return FTTransformerRegressor
        case 'dfttc':
            return DiffFTTransformerClassifier
        case 'dfttr':
            return DiffFTTransformerRegressor
