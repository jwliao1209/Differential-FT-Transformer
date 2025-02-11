import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from trainer import Trainer


class DummyModel(nn.Module):
    """Dummy model for testing Trainer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X, y=None):
        output = self.linear(X)
        loss = None
        if y is not None:
            loss = torch.nn.functional.mse_loss(output, y.float())
        return {"pred": output, "loss": loss}
    
    def predict(self, X):
        return self.forward(X)["pred"].argmax(dim=1)


@pytest.fixture
def setup_trainer():
    """Fixture to set up Trainer with a dummy model."""
    input_dim = 10
    output_dim = 1
    model = DummyModel(input_dim, output_dim)
    batch_size = 16
    n_epoch = 5
    lr = 1e-4
    weight_decay = 1e-5
    eval_funs = {
        "r2": r2_score,
    }
    trainer = Trainer(
        model=model,
        batch_size=batch_size,
        n_epoch=n_epoch,
        lr=lr,
        weight_decay=weight_decay,
        eval_funs=eval_funs,
    )
    return trainer


def test_set_random_seed(setup_trainer):
    """Test setting random seed."""
    trainer = setup_trainer
    trainer.set_random_seed(random_state=42)
    assert torch.initial_seed() == 42


def test_set_optimization(setup_trainer):
    """Test optimizer setup."""
    trainer = setup_trainer
    trainer.set_optimization()
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]['lr'] == trainer.lr
    assert trainer.optimizer.param_groups[0]['weight_decay'] == trainer.weight_decay


def test_train(setup_trainer):
    """Test the training loop."""
    trainer = setup_trainer
    input_dim = 10
    train_X = torch.randn(32, input_dim)
    train_y = torch.randn(32, 1)
    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=trainer.batch_size,
        shuffle=True
    )
    initial_weight = trainer.model.linear.weight.clone()
    trainer.train(train_loader)
    # Check if model weights are updated
    assert not torch.equal(initial_weight, trainer.model.linear.weight)


def test_test(setup_trainer):
    """Test the evaluation loop."""
    trainer = setup_trainer
    input_dim = 10
    test_X = torch.randn(32, input_dim)
    test_y = torch.randn(32, 1)
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=trainer.batch_size,
        shuffle=False
    )
    result = trainer.test(test_loader)
    assert "r2" in result
    assert result["r2"] <= 1.0


def test_fit(setup_trainer):
    """Test the fit method."""
    trainer = setup_trainer
    input_dim = 10
    train_X = torch.randn(64, input_dim)
    train_y = torch.randn(64, 1)
    test_X = torch.randn(32, input_dim)
    test_y = torch.randn(32, 1)

    trainer.fit(train_X, train_y, test_X=test_X, test_y=test_y)

    # Check if model weights are updated during training
    initial_weight = trainer.model.linear.weight.clone()
    trainer.train(DataLoader(TensorDataset(train_X, train_y), batch_size=trainer.batch_size))
    assert not torch.equal(initial_weight, trainer.model.linear.weight)
