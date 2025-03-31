import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text_batch, label_batch in tqdm(train_loader, desc="Training"):
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)

        optimizer.zero_grad()
        predictions = model(text_batch)
        loss = criterion(predictions, label_batch)

        acc = (predictions.argmax(1) == label_batch).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for text_batch, label_batch in tqdm(test_loader, desc="Evaluating"):
            text_batch, label_batch = text_batch.to(device), label_batch.to(device)
            predictions = model(text_batch)
            loss = criterion(predictions, label_batch)
            acc = (predictions.argmax(1) == label_batch).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(test_loader), total_acc / len(test_loader)
