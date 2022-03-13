import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import BERTDiagnosisNER

def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def trainer(train_loader: DataLoader, val_loader: DataLoader, model: BERTDiagnosisNER, config: dict, device: str = "cuda"):
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparams"])

    # some variables
    record = {
        "dx": {"acc": list(), "loss": list()},
        "ner": {"acc": list(), "loss": list()}
    }
    best_dx_acc = 0
    best_ner_acc = 0
    step = 0

    for epoch in range(1, config["n_epochs"] + 1):
        for x, y_dx, y_ner in train_loader:
            model.train()
            # move data to device
            x = move_bert_input_to_device(x, device)
            y_dx = y_dx.to(device)
            y_ner = y_ner.to(device)
            # make prediction and calculate loss
            o_dx, o_ner = model(x)
            _, _, total_loss = model.calc_loss(o_dx, y_dx, o_ner, y_ner)

            # back-propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # evaluate model every k steps
            if step % config["checkpoint_steps"] == 0:
                print("Evaluating model at step {}...".format(step))
                record, best_dx_acc, best_ner_acc = update_evaluation(val_loader, model, config, device, record, best_dx_acc, best_ner_acc)
            step += 1
        
        # evaluate model every epoch
        print("===== Evaluting model at epoch {} =====".format(epoch))
        record, best_dx_acc, best_ner_acc = update_evaluation(val_loader, model, config, device, record, best_dx_acc, best_ner_acc)
    
    record["best_dx_acc"] = best_dx_acc
    record["best_ner_acc"] = best_ner_acc
    return record

def update_evaluation(data_loader, model, config, device, record, best_dx_acc, best_ner_acc):
    # utility function
    def update_record(record, dx_acc, dx_loss, ner_acc, ner_loss):
        record["dx"]["acc"].append(dx_acc)
        record["dx"]["loss"].append(dx_loss)
        record["ner"]["acc"].append(ner_acc)
        record["ner"]["loss"].append(ner_loss)
        return record
    # update metrics
    dx_acc, ner_acc = evaluate_model_acc(data_loader, model, device)
    dx_loss, ner_loss = evaluate_model_loss(data_loader, model, device)
    record = update_record(record, dx_acc, dx_loss, ner_acc, ner_loss)
    print(f"Diagnosis: acc -> {dx_acc:.4f}; loss -> {dx_loss:.4f} / NER: acc -> {ner_acc:.4f}; loss -> {ner_loss:.4f}")
    if dx_acc > best_dx_acc:
        best_dx_acc = dx_acc
        # save model
        if config["model_save_name"]:
            torch.save(model.state_dict(), "./models/{}.pth".format(config["model_save_name"]))
            print("Best model saved.")
    if ner_acc > best_ner_acc:
        best_ner_acc = ner_acc

    return record, best_dx_acc, best_ner_acc

def evaluate_model_acc(data_loader, model, device):
    dx_correct, dx_predict, ner_correct, ner_predict = 0, 0, 0, 0

    model.eval()
    for x, y_dx, y_ner in data_loader:
        x = move_bert_input_to_device(x, device)
        y_dx = y_dx.to(device)
        y_ner = y_ner.to(device)
        with torch.no_grad():
            o_dx, o_ner = model(x)
            # dx acc
            dx_correct += (o_dx.argmax(dim=-1) == y_dx).sum().cpu().detach().item()
            dx_predict += len(y_dx)
            # ner acc
            ner_correct += (o_ner.argmax(dim=-1) == y_ner).sum().cpu().detach().item()
            ner_predict += (y_ner != model.ner_ignore_index).sum().cpu().detach().item()
    
    dx_acc = dx_correct / dx_predict
    ner_acc = ner_correct / ner_predict
    return dx_acc, ner_acc

def evaluate_model_loss(data_loader, model, device):
    total_dx_loss, total_ner_loss = 0.0, 0.0

    model.eval()
    for x, y_dx, y_ner in data_loader:
        # move data to device
        x = move_bert_input_to_device(x, device)
        y_dx = y_dx.to(device)
        y_ner = y_ner.to(device)
        with torch.no_grad():
            o_dx, o_ner = model(x)
            dx_loss, ner_loss, _ = model.calc_loss(o_dx, y_dx, o_ner, y_ner)
        # accumulate loss
        total_dx_loss += dx_loss.cpu().detach().item() * y_dx.shape[0]
        total_ner_loss += ner_loss.cpu().detach().item() * y_ner.shape[0]
    
    mean_dx_loss = total_dx_loss / len(data_loader.dataset)
    mean_ner_loss = total_ner_loss / len(data_loader.dataset)
    return mean_dx_loss, mean_ner_loss

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
        