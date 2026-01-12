import torch
from mlops_26.model import MyAwesomeModel

def test_dimensions():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)  # batch of 4 images
    output = model(x)
    assert output.shape == (1, 10)  # batch of 4 outputs for 10 classes


# tests/test_model.py
import pytest
from mlops_26.model import MyAwesomeModel

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match=r"Expected each sample to have shape \[1, 28, 28\]"):
        model(torch.randn(1,1,28,29))

# We can also test with parametrized test, where we can test for multiple input values within the same test function
@pytest.mark.parametrize("batch_size", [32,64])
def test_model(batch_size):
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)


