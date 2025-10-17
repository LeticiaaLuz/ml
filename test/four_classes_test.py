"""Simple trainer in 4 classes dataset
"""
import argparse

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty

import lps_ml.models.mlp as lps_mlp
import lps_ml.utils as lps_utils
import lps_ml.databases.cv as lps_cv
import lps_ml.processors as lps_proc
import lps_ml.databases.four_classes as lps_4classes

def _evaluate_accuracy(model: torch.nn.Module,
                      dataloader: torch_data.DataLoader):
    device = lps_utils.get_available_device()
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            if out.ndim == 1:
                preds = (out > 0.5).long()
            else:
                preds = torch.argmax(out, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return acc

def _main():

    parser = argparse.ArgumentParser(description="Train an MLP classifier on MNIST.")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Directory to store MNIST data.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    # parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp",
    #                     help="Model type to train: 'mlp' or 'cnn'.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')

    fs_out=lps_qty.Frequency.khz(16)
    duration=lps_qty.Time.s(1)
    overlap=lps_qty.Time.s(0)

    dm = lps_4classes.FourClasses(
            file_processor=lps_proc.WindowingResampler(fs_out=fs_out,
                                                        duration=duration,
                                                        overlap=overlap),
            cv = lps_cv.FiveByTwo(),
            batch_size=8)

    n_targets = 4

    model = lps_mlp.MLP(
        input_shape=(1, int(fs_out*duration)),
        hidden_channels=[64, 16],
        n_targets=n_targets,
        loss_fn=torch.nn.CrossEntropyLoss,
        lr=args.lr,
    )

    checkpoint_cb = lightning_call.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename=f"mnist-{{epoch:02d}}-{{val_loss:.3f}}",
    )
    early_stop_cb = lightning_call.EarlyStopping(monitor="val_loss", patience=4, mode="min")

    logger = lightning_log.TensorBoardLogger(
        "logs",
        name="mnist"
    )

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    train_acc = _evaluate_accuracy(model, dm.train_dataloader())
    val_acc   = _evaluate_accuracy(model, dm.val_dataloader())
    # test_acc  = _evaluate_accuracy(model, dm.test_dataloader())

    print(f"{'='*60}")
    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    # print(f"Test accuracy:       {test_acc:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    _main()
