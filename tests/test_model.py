import pytest
import torch

from models.ft_transformer import FTTransformer, FTTransformerClassifier, FTTransformerRegressor


@pytest.fixture
def setup_transformer():
    """Fixture to set up a FTTransformer instance."""
    n_class = 3  # For classification
    category_column_count = [-1, -1, 3, 4, -1]  # -1 for numerical, integers for categorical
    d_token = 192
    transformer = FTTransformer(
        n_class=n_class,
        category_column_count=category_column_count,
        d_token=d_token,
        d_ffn_factor=4 / 3,
        n_layer=3,
        n_head=8,
        attention_dropout_rate=0.2,
        ffn_dropout_rate=0.1,
        residual_dropout_rate=0.,
        use_bias=True,
    )
    return transformer


def test_initialization(setup_transformer):
    """Test FTTransformer initialization."""
    transformer = setup_transformer
    assert transformer.n_class == 3
    assert transformer.num_tokens > 0
    assert len(transformer.layers) == 3
    assert isinstance(transformer.head, torch.nn.Linear)
    assert isinstance(transformer.feat_tokenizer, torch.nn.Module)


def test_forward_classification(setup_transformer):
    """Test forward pass for classification."""
    n_class = 3
    category_column_count = [-1, -1, 3, 4, -1]
    d_token = 192
    transformer = FTTransformerClassifier(
        n_class=n_class,
        category_column_count=category_column_count,
        d_token=d_token,
        d_ffn_factor=4 / 3,
        n_layer=3,
        n_head=8,
        attention_dropout_rate=0.2,
        ffn_dropout_rate=0.1,
        residual_dropout_rate=0.,
        use_bias=True,
    )
    batch_size = 4
    seq_len = 5
    x = torch.rand(batch_size, seq_len)
    y = torch.randint(0, 3, (batch_size,))  # For classification with 3 classes

    output = transformer(x, y)
    assert "pred" in output
    assert "loss" in output
    assert output["pred"].shape == (batch_size, 3)  # Output size should match n_class
    assert output["loss"].item() >= 0  # Loss should be a non-negative scalar


def test_forward_regression():
    """Test forward pass for regression."""
    category_column_count = [-1, -1, 3, 4, -1]
    d_token = 192
    transformer = FTTransformerRegressor(
        category_column_count=category_column_count,
        d_token=d_token,
        d_ffn_factor=4 / 3,
        n_layer=3,
        n_head=8,
        attention_dropout_rate=0.2,
        ffn_dropout_rate=0.1,
        residual_dropout_rate=0.,
        use_bias=True,
    )

    batch_size = 4
    seq_len = 5
    x = torch.rand(batch_size, seq_len)
    y = torch.rand(batch_size, 1)  # Regression targets

    output = transformer(x, y)
    assert "pred" in output
    assert "loss" in output
    assert output["pred"].shape == (batch_size, 1)  # Output size should match n_class
    assert output["loss"].item() >= 0  # Loss should be a non-negative scalar


def test_feature_tokenizer_integration(setup_transformer):
    """Test that the FeatureTokenizer correctly integrates with FTTransformer."""
    transformer = setup_transformer
    batch_size = 4
    seq_len = 5
    x = torch.rand(batch_size, seq_len)

    # Ensure that the tokenizer produces valid embeddings
    tokenized = transformer.feat_tokenizer(x)
    assert tokenized.shape == (batch_size, transformer.num_tokens, transformer.feat_tokenizer.d)


def test_attention_weights_dimension(setup_transformer):
    """Test the attention weights and outputs for correct dimensions."""
    transformer = setup_transformer
    batch_size = 4
    seq_len = 5
    x = torch.rand(batch_size, seq_len)

    output = transformer(x)
    pred = output["pred"]
    assert pred.shape == (batch_size, 3)  # Output size should match n_class
