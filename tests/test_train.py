from __future__ import annotations

import torch


def test_train_runs_and_saves_checkpoint(tmp_path, monkeypatch):
    # Run inside temp folder so your repo doesn't get new files
    monkeypatch.chdir(tmp_path)

    import mlops_26.train as train_module

    # Make training fast by patching the dataset to be tiny
    def fake_corrupt_mnist():
        x = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        ds = torch.utils.data.TensorDataset(x, y)
        return ds, ds

    monkeypatch.setattr(train_module, "corrupt_mnist", fake_corrupt_mnist)

    # Run minimal training (1 epoch)
    train_module.train(lr=1e-3, batch_size=32, epochs=1)

    # Check outputs were created
    assert (tmp_path / "models" / "model.pth").exists()
    assert (tmp_path / "reports" / "figures" / "training_statistics.png").exists()

    # Check checkpoint is loadable
    state = torch.load(tmp_path / "models" / "model.pth", map_location="cpu")
    assert isinstance(state, dict)
    assert len(state) > 0
