import pytest
import torch

from models.layers import FeatureTokenizer


@pytest.fixture
def setup_tokenizer():
    category_column_count = [-1, -1, 3, 4, -1]  # -1 for numerical, integers for categorical
    d = 8  # Embedding dimension
    use_bias = True
    tokenizer = FeatureTokenizer(category_column_count, d, use_bias)
    return tokenizer, category_column_count


def test_extract_feature_metadata(setup_tokenizer):
    tokenizer, category_column_count = setup_tokenizer

    # Expected output
    expected_numerical_index = [0, 1, 4]
    expected_categorical_index = [2, 3]
    expected_categorical_count = [3, 4]
    expected_categorical_offset = torch.tensor([0, 3])

    # Call method
    index_info = tokenizer.extract_feature_metadata(category_column_count)

    # Assert results
    assert index_info['numerical_index'] == expected_numerical_index
    assert index_info['categorical_index'] == expected_categorical_index
    assert index_info['categorical_count'] == expected_categorical_count
    torch.testing.assert_close(index_info['categorical_offset'], expected_categorical_offset)


def test_layers_initialization(setup_tokenizer):
    tokenizer, _ = setup_tokenizer

    # Assert numerical layer
    numerical_layer = tokenizer.layers['numerical']
    assert numerical_layer.size() == torch.Size([len(tokenizer.numerical_index) + 1, tokenizer.d])

    # Assert categorical layer
    categorical_layer = tokenizer.layers['categorical']
    assert categorical_layer.weight.size() == torch.Size([sum(tokenizer.categorical_count), tokenizer.d])

    # Assert bias layer
    if tokenizer.use_bias:
        bias_layer = tokenizer.layers['bias']
        assert bias_layer.size() == torch.Size([tokenizer.num_feat, tokenizer.d])


def test_forward(setup_tokenizer):
    tokenizer, _ = setup_tokenizer

    # Mock input
    x = torch.tensor([
        [1.0, 2.0, 0, 1, 3],
        [4.0, 5.0, 2, 3, 6]
    ])  # Numerical: [0, 1, 4], Categorical: [2, 3]

    # Call forward
    output = tokenizer(x)

    # Assert output shape
    batch_size = x.size(0)
    expected_features = len(tokenizer.numerical_index) + len(tokenizer.categorical_index) + 1
    assert output.size() == torch.Size([batch_size, expected_features, tokenizer.d])

    # Assert CLS token embedding
    cls_embedding = output[:, 0, :]
    assert (cls_embedding != 0).all()

    # Assert categorical features embedding
    cat_embedding = output[:, len(tokenizer.numerical_index) + 1:, :]
    assert cat_embedding.size(1) == len(tokenizer.categorical_index)


def test_no_bias():
    category_column_count = [-1, -1, 3, 4, -1]
    d = 8
    tokenizer = FeatureTokenizer(category_column_count, d, use_bias=False)

    # Mock input
    x = torch.tensor([
        [1.0, 2.0, 0, 1, 3],
        [4.0, 5.0, 2, 3, 6]
    ])

    # Call forward
    output = tokenizer(x)

    # Assert output shape
    batch_size = x.size(0)
    expected_features = len(tokenizer.numerical_index) + len(tokenizer.categorical_index) + 1
    assert output.size() == torch.Size([batch_size, expected_features, d])

    # Assert no bias in layers
    assert 'bias' not in tokenizer.layers
