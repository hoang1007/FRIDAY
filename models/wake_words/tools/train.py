import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from typing import Dict
import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

from tqdm import tqdm
from argparse import ArgumentParser

from wake_words.configs import exp1 as config
from wake_words.src.model import WakeWordDetector
from wake_words.src.dataset import WakeWordDataConstructor


def main(args):
    data_constructor = WakeWordDataConstructor(**config.data)
    model = WakeWordDetector(**config.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.trainer.get('lr', 1e-3),
        weight_decay=config.trainer.get('weight_decay', 0)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    metrics: Dict[str, Metric] = {
        "accuracy": BinaryAccuracy(),
        "micro_f1_score": BinaryF1Score(multidim_average='global'),
        "confusion_matrix": BinaryConfusionMatrix(),
    }

    model.eval()
    for batch_idx, batch in enumerate(data_constructor.val_dataloader()):
        waveforms, labels = batch
        predicted, _ = model(waveforms)
        for metric in metrics.values():
            metric.update(predicted, labels)
        if batch_idx > 5:
            break
    for name, metric in metrics.items():
        print(f"{name}: {metric.compute()}")
        metric.reset()

    for epoch in range(config.trainer.get('epochs', 1)):
        model.train()
        with tqdm(data_constructor.train_dataloader()) as pbar:
            for batch_idx, batch in enumerate(pbar):
                pbar.set_description(f"Epoch {epoch}")
                waveforms, labels = batch
                loss = model(waveforms, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:
                    pbar.set_postfix(loss=loss.item())
        scheduler.step()
        model.eval()
        with tqdm(data_constructor.val_dataloader()) as pbar:
            for batch_idx, batch in enumerate(pbar):
                pbar.set_description(f"Epoch {epoch}")
                waveforms, labels = batch
                predicted, _ = model(waveforms)
                for metric in metrics.values():
                    metric.update(predicted, labels)
        for name, metric in metrics.items():
            print(f"{name}: {metric.compute()}")
            metric.reset()
        ckpt_path = os.path.join(config.trainer.get('ckpt_dir'), f'epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    main(None)
