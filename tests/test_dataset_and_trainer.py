"""
Unit tests for dataset loading, stratified splitting, and the training pipeline.
Runs a 2-epoch miniature training cycle to verify checkpoint creation, metric computation, and evaluation.
"""

import os
import shutil
import tempfile
import torch
from datasets import load_enzymes_dataset, get_stratified_splits, get_dataloaders
from models import ProteinGATv2
from augmentations import bio_aga_edge_drop
from training.trainer import ProteinTrainer


def test_training_pipeline_mini():
    dataset = load_enzymes_dataset(root="data/raw", use_node_attr=True)
    assert len(dataset) == 600

    labels = [int(data.y.item()) for data in dataset]
    splits = get_stratified_splits(labels, num_folds=5, seed=42)
    split_0 = splits[0]

    train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=16)

    device = torch.device("cpu")
    model = ProteinGATv2(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=16,
        heads=2,
        num_layers=2,
        dropout=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    temp_dir = tempfile.mkdtemp()
    try:
        trainer = ProteinTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            regularizer_fn=lambda b: bio_aga_edge_drop(b, drop_prob=0.2, gamma=0.35),
            checkpoint_dir=temp_dir
        )

        res = trainer.fit(train_loader, val_loader, test_loader, epochs=2, patience=2, model_name="test_chk.pt")

        assert "test_metrics" in res
        assert "accuracy" in res["test_metrics"]
        assert "macro_f1" in res["test_metrics"]
        assert os.path.exists(os.path.join(temp_dir, "test_chk.pt"))
    finally:
        shutil.rmtree(temp_dir)
