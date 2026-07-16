import torch

def train_one_epoch(model, train_loader, optimizer, loss_fn, device='cpu'):
    
    epoch_loss = 0.0

    for batch_id, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model(inputs)

        optimizer.zero_grad(set_to_none=True)

        loss = loss_fn(pred, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

def eval_one_epoch(model, val_loader, loss_fn, device='cpu'):

    epoch_loss = 0.0

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(val_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model(inputs)

            loss = loss_fn(pred, labels)

            epoch_loss += loss.item()

    return epoch_loss / len(val_loader)





