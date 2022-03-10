import torch
from tqdm import tqdm

def trainer(train_loader, val_loader, model, criterion, config, device):
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    # calculate initial val loss
    val_loss = evaluate_model_loss(val_loader, model, criterion, device)
    print(f"Initial val loss: {val_loss:.4f}")

    # some variables
    loss_record = {"train": list(), "val": list()}
    min_loss = float("inf")
    step = 0

    for epoch in range(1, config["n_epochs"] + 1):
        # train
        model.train()
        for x, y in train_loader:
            # move data to device
            move_bert_input_to_device(x, device)
            y = y.to(device)
            # make prediction and calculate loss
            pred = model(x)
            loss = criterion(pred, y)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss
            loss_record["train"].append(loss.detach().cpu().item())
            if step % 100 == 0:
                print(f"Step {step}: batch_train_loss = {loss_record['train'][-1]}")
            step += 1
        
        # evaluate
        val_loss = evaluate_model_loss(val_loader, model, criterion, device)
        loss_record["val"].append(val_loss)
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), config["model_save_path"])
            print(f"Best model saved (epoch = {epoch:2d}, loss = {min_loss:.4f})")
    
    return min_loss, loss_record


def evaluate_model_loss(data_loader, model, criterion, device):
    model.eval()
    total_val_loss = 0

    for x, y in data_loader:
        # move data to device
        move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
        total_val_loss += loss.detach().cpu().item() * len(x["input_ids"])    
    
    mean_val_loss = total_val_loss / len(data_loader.dataset)
    return mean_val_loss

def predict_whole_set(model, data_loader, device) -> list:
    preds = list()
    model.eval()
    print(f"Predicting set...")
    for x, y in tqdm(data_loader):
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    return torch.cat(preds)

def move_bert_input_to_device(x, device):
    for k in x:
        x[k] = x[k].to(device)
    return x
        