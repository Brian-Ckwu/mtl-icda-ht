import torch
from tqdm import tqdm
from colorama import Fore, Style

def trainer(train_loader, val_loader, model, criterion, config, device):
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    # calculate initial val loss
    initial_val_loss = evaluate_model_loss(val_loader, model, criterion, device)
    print(f"Initial val loss: {initial_val_loss:.4f}")

    # some variables
    loss_record = {"train": list(), "val": list()}
    min_loss = float("inf")

    for epoch in range(1, config["n_epochs"] + 1):
        # train
        total_train_loss = 0
        model.train()
        print("Training epoch {}:".format(epoch))
        for x, y in tqdm(train_loader):
            # move data to device
            move_bert_input_to_device(x, device)
            y = y.to(device)
            # make prediction and calculate loss
            pred = model(x).transpose(1, 2)
            loss = criterion(pred, y)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # accumulate loss
            total_train_loss += loss.detach().cpu().item() * len(x["input_ids"])
        # record train loss
        epoch_train_loss = total_train_loss / len(train_loader.dataset)
        loss_record["train"].append(epoch_train_loss)
        
        # evaluate
        epoch_val_loss = evaluate_model_loss(val_loader, model, criterion, device)
        loss_record["val"].append(epoch_val_loss)

        # Print loss
        print(f"Finish training epoch {epoch}: train loss - {epoch_train_loss:.4f}, val loss - {epoch_val_loss:.4f}")

        if epoch_val_loss < min_loss:
            min_loss = epoch_val_loss
            torch.save(model.state_dict(), config["model_save_path"])
            print(f"Best model saved (epoch = {epoch:2d}, loss = {min_loss:.4f})")
    
    return loss_record


def evaluate_model_loss(data_loader, model, criterion, device):
    model.eval()
    total_val_loss = 0

    for x, y in tqdm(data_loader):
        # move data to device
        move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x).transpose(1, 2)
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
        
def visualize_ner_labels(tokenizer, input_ids, ner_labels):
    for i, token_id in enumerate(input_ids[0]):
        token = tokenizer.decode(token_id)
        if token[:2] == "##":
            token = token[2:]
            print('\b', end='')
        if ner_labels[i] == 0:
            print(Style.RESET_ALL + token, end=' ')
        else:
            print(Fore.RED + token, end=' ')

def unsqueeze_input(x):
    for key in x.keys():
        x[key] = torch.unsqueeze(x[key], dim=0)
    return x