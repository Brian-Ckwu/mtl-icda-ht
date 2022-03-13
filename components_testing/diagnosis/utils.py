import random
import numpy as np
import torch
from tqdm import tqdm

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
            pred = model(x)
            loss = criterion(pred, y)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # evaluate model every k steps
            if step % config["checkpoint_steps"] == 0:
                print("Evaluating model at step {}...".format(step))
                best_val_acc = update_evaluation(val_loader, model, criterion, config, device, record, best_val_acc)
            step += 1
        
        # evaluate model every epoch
        print("===== Evaluting model at epoch {} =====".format(epoch))
        best_val_acc = update_evaluation(val_loader, model, criterion, config, device, record, best_val_acc)
    
    record["best_val_acc"] = best_val_acc
    return record

def update_evaluation(data_loader, model, criterion, config, device, record, best_acc):
    acc = evaluate_model_acc(data_loader, model, device)
    loss = evaluate_model_loss(data_loader, model, criterion, device)
    record["acc"].append(acc)
    record["loss"].append(loss)
    print(f"Acc: {acc:.4f} / Loss: {loss:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "./models/{}.pth".format(config["model_save_name"]))
        print("Best model saved at this checkpoint.")

    return best_acc

def evaluate_model_acc(data_loader, model, device):
    model.eval()
    n_correct = 0
    n_predict = 0

    for x, y in data_loader:
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
            n_correct += (pred.argmax(dim=-1) == y).sum().cpu().detach().item()
            n_predict += len(y)
    
    acc = n_correct / n_predict
    return acc

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
        total_val_loss += loss.detach().cpu().item() * len(y)
    
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
        