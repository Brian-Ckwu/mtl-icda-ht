import torch
import random
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def trainer(train_loader, val_loader, model, criterion, config, device):
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    # some variables
    record = {
        "acc": list(),
        "loss": list()
    }
    best_val_acc = 0
    step = 0

    for epoch in range(1, config["n_epochs"] + 1):
        for x, y in train_loader:
            model.train()
            # move data to device
            x = move_bert_input_to_device(x, device)
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
                print("Evaluating model at step {}...".format(step))
                best_val_acc = update_evaluation(val_loader, model, criterion, config, device, record, best_val_acc)            
            step += 1

        # evaluate model at the end of one epoch
        print("===== Evaluating model at epoch {} =====".format(epoch))
        best_val_acc = update_evaluation(val_loader, model, criterion, config, device, record, best_val_acc)

    record["best_val_acc"] = best_val_acc
    return record

def update_evaluation(data_loader, model, criterion, config, device, record, best_acc):
    # utility function
    def update_record(record, acc, loss):
        record["acc"].append(acc)
        record["loss"].append(loss)
        return record
    # update metrics
    acc = evaluate_model_acc(data_loader, model, device)
    loss = evaluate_model_loss(data_loader, model, criterion, device)
    record = update_record(record, acc, loss)
    print(f"Acc: {acc:.4f} / Loss: {loss:.4f}")
    if acc > best_acc:
        best_acc = acc
        if config["model_save_name"]:
            torch.save(model.state_dict(), "./models/{}.pth".format(config["model_save_name"]))
            print("Best model saved.")

    return best_acc

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
        total_val_loss += loss.detach().cpu().item() * y.shape[0]
    
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