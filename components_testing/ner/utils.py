import torch
from tqdm import tqdm
from colorama import Fore, Style

def trainer(train_loader, val_loader, model, criterion, config, device):
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    # some variables
    record = {
        "train": {"loss": list(), "acc": list()},
        "val": {"loss": list(), "acc": list()}
    }
    best_val_acc = 0
    step = 0

    for epoch in range(1, config["n_epochs"] + 1):
        print("Training epoch {}:".format(epoch))
        for x, y in tqdm(train_loader):
            model.train()
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
            
            # evaluate model at the checkpoint step
            if step % config["checkpoint_steps"] == 0:
                val_acc = evaluate_model_acc(val_loader, model, device)
                val_loss = evaluate_model_loss(val_loader, model, criterion, device)
                record["val"]["acc"].append(val_acc)
                record["val"]["loss"].append(val_loss)
                print(f"Step {step}: val acc -> {val_acc:.4f}; val loss -> {val_loss:.4f}")
                # save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), config["model_save_path"])
                    print(f"Best model saved (step = {step}, acc = {val_acc:.4f})")
            
            step += 1

        
        # evaluate model at the end of one epoch
        epoch_val_acc = evaluate_model_acc(val_loader, model, device)
        epoch_val_loss = evaluate_model_loss(val_loader, model, criterion, device)
        record["val"]["acc"].append(epoch_val_acc)
        record["val"]["loss"].append(epoch_val_loss)

        # Print evaluation metrics
        print(f"\n ===== Finish training epoch {epoch}: val acc -> {epoch_val_acc:.4f}; val loss -> {epoch_val_loss:.4f} =====\n")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), config["model_save_path"])
            print(f"Best model saved (epoch = {epoch:2d}, acc = {best_val_acc:.4f})")
    
    return record


def evaluate_model_loss(data_loader, model, criterion, device):
    model.eval()
    total_val_loss = 0

    for x, y in data_loader:
        # move data to device
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x).transpose(1, 2) # transpose for calculating cross entropy loss
            loss = criterion(pred, y)
        total_val_loss += loss.detach().cpu().item() * len(x["input_ids"])
    
    mean_val_loss = total_val_loss / len(data_loader.dataset)
    return mean_val_loss

def evaluate_model_acc(data_loader, model, device):
    total_tokens = 0
    total_correct = 0
    
    model.eval()
    for x, y in data_loader:
        # inference
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x) # no need to transpose in this case
        # calculate target metric (acc)
        total_tokens += (y != -100).sum().cpu().item()
        total_correct += (pred.argmax(dim=-1) == y).sum().cpu().item()
    
    acc = total_correct / total_tokens
    return acc

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