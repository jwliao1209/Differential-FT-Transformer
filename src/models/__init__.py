from typing import Union

from .ft_transformer import FTTransformer, FTTransformerClassifier, FTTransformerRegressor
from .diff_ft_transformer import DiffFTTransformer, DiffFTTransformerClassifier, DiffFTTransformerRegressor
from .dint_ft_transformer import DINTFTTransformer, DINTFTTransformerClassifier, DINTFTTransformerRegressor
from .dofen import DOFENClassifier, DOFENRegressor


def get_model(model_name: str) -> Union[FTTransformer, DiffFTTransformer, DINTFTTransformer]:
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
            return DINTFTTransformerClassifier
        case 'dintr':
            return DINTFTTransformerRegressor
        case 'dofenc':
            return DOFENClassifier
        case 'dofenr':
            return DOFENRegressor
