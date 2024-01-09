from .loss_fn import negative_feedback
from tqdm import tqdm


def train_fn(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def train_fn_irs(args, model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = negative_feedback(outputs[-1], target, args.beta, criterion, outputs[:-1])
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def train_fn_ovf(args, model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        model.noise_backbone = args.var1 + 0.15
        ovf1 = model(data)
        model.noise_backbone = args.var1 + 0.1
        ovf2 = model(data)
        model.noise_backbone = args.var1 + 0.05
        ovf3 = model(data)
        model.noise_backbone = args.var1
        loss = negative_feedback(output, target, args.beta, criterion, [ovf1, ovf2, ovf3])
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss