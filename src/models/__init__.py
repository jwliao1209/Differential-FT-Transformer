from typing import Dict, Union

from .ft_transformer import FTTransformer, FTTransformerClassifier, FTTransformerRegressor
from .diff_ft_transformer import DiffFTTransformer, DiffFTTransformerClassifier, DiffFTTransformerRegressor
from .dint_ft_transformer import DINTFTTransformer, DINTFTTransformerClassifier, DINTFTTransformerRegressor
from .dofen import DOFENClassifier, DOFENRegressor
from .dofen_transformer import DOFENTransformerClassifier, DOFENTransformerRegressor
from .dyt import convert_ln_to_dyt, convert_ln_to_dyat, convert_ln_to_fdyat, convert_ln_to_dys, convert_ln_to_dyas


def get_model(
    model_name: str,
    model_config: Dict[str, Union[int, float, str]],
) -> Union[FTTransformer, DiffFTTransformer, DINTFTTransformer]:

    match model_name:
        case 'ftc':
            model = FTTransformerClassifier(**model_config)
        case 'ftr':
            model = FTTransformerRegressor(**model_config)
        case 'diffc':
            model = DiffFTTransformerClassifier(**model_config)
        case 'diffr':
            model = DiffFTTransformerRegressor(**model_config)
        case 'dintc':
            model = DINTFTTransformerClassifier(**model_config)
        case 'dintr':
            model = DINTFTTransformerRegressor(**model_config)
        case 'dofenc':
            model = DOFENClassifier(**model_config)
        case 'dofenr':
            model = DOFENRegressor(**model_config)
        case 'doformerc':
            model = DOFENTransformerClassifier(**model_config)
        case 'doformerr':
            model = DOFENTransformerRegressor(**model_config)
        case _:
            raise ValueError(f"Invalid model name: {model_name}")

    match model_config['norm']:
        case 'layer_norm':
            pass
        case 'dyt':
            model = convert_ln_to_dyt(model)
        case 'dyat':
            model = convert_ln_to_dyat(model)
        case 'fdyat':
            model = convert_ln_to_fdyat(model)
        case 'dys':
            model = convert_ln_to_dys(model)
        case 'dyas':
            model = convert_ln_to_dyas(model)
        case _:
            raise ValueError(f"Invalid norm: {model_config['norm']}")

    return model
