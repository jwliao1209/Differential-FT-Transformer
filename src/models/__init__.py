from typing import Union

from .ft_transformer import FTTransformer, FTTransformerClassifier, FTTransformerRegressor
from .diff_ft_transformer import DiffFTTransformer, DiffFTTransformerClassifier, DiffFTTransformerRegressor
from .dint_ft_transformer import DintFTTransformer, DintFTTransformerClassifier, DintFTTransformerRegressor


def get_model(model_name: str) -> Union[FTTransformer, DiffFTTransformer, DintFTTransformer]:
    match model_name:
        case 'ftc':
            return FTTransformerClassifier
        case 'ftr':
            return FTTransformerRegressor
        case 'diffc':
            return DiffFTTransformerClassifier
        case 'diffr':
            return DiffFTTransformerRegressor
        case 'dintc':
            return DintFTTransformerClassifier
        case 'dintr':
            return DintFTTransformerRegressor
