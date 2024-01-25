import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from text_module.text_embedding import TextEmbedding
from load_data.data_loader import Dataset, getDataloader
from models.model import PickEvidenceIndexModel

def train(model, text_embedding, train_loader, criterion, optimizer, device, epoch):
    model.train()
    avg_loss = 0
    total_true_pred = 0
    total_sample = 0

    for batch in tqdm(train_loader, desc=f"TRAINING EPOCH {epoch}"):
        claim, context, label = batch["claim"], batch["context"], batch["label"]
        
        #text_embedding
        input_ids = text_embedding(claim, context).to(device)

        #forward call
        out, out_soft = model(input_ids)
        loss = criterion(out.to(device), label.to(device))
        avg_loss += loss

        #calc acc
        total_true_pred += (out_soft.argmax(dim=-1).to(device) == label.to(device)).sum()
        total_sample += label.size()[0]

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss/len(train_loader), total_true_pred/total_sample

def eval(model, text_embedding, eval_loader, criterion, device, epoch):
    model.eval()
    avg_loss = 0
    total_true_pred = 0
    total_sample = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"EVALUATING EPOCH {epoch}"):
            claim, context, label = batch["claim"], batch["context"], batch["label"]

            #text_embedding
            input_ids = text_embedding(claim, context).to(device)

            #forward call
            out, out_soft = model(input_ids)
            loss = criterion(out.to(device), label.to(device))
            avg_loss += loss

            #calc acc
            total_true_pred += (out_soft.argmax(dim=-1).to(device) == label.to(device)).sum()
            total_sample += label.size()[0]
    return avg_loss/len(eval_loader), total_true_pred/total_sample

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("💩💩💩💩💩DEVICE LÀ:", device)

    #khởi tạo text_embedding
    text_embedding = TextEmbedding(config).to(device)
    print("💩💩💩💩💩KHỞI TẠO TEXT EMBEDDING XONG.")

    #load data
    train_dataset = Dataset(config["data_path"]["train"])
    train_loader = getDataloader(train_dataset, config["train"]["batch_size"])
    #eval_dataset = Dataset(config["data_path"]["eval"])
    #eval_loader = getDataloader(eval_dataset, config["train"]["batch_size"])
    print("💩💩💩💩💩LOAD DATA THÀNH CÔNG.")
    
    #khởi tạo model
    model = PickEvidenceIndexModel(config).to(device)
    print("💩💩💩💩💩KHỞI TẠO MODEL THÀNH CÔNG.")

    #cài đặt optim và criterion
    optimizer = optim.AdamW(model.parameters(), config["train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    #train
    for epoch in range(config["train"]["num_epochs"]):
        avg_loss, acc = train(model, text_embedding, train_loader, criterion, optimizer, device, epoch + 1)
        #avg_loss_eval, acc_eval = eval(model, text_embedding, eval_loader, criterion, device, epoch + 1)
        print("💩💩💩💩💩-Avg loss: {:.4f} 💩💩💩💩💩-Accuracy: {:.2f}%".format(avg_loss.item(), acc.item()*100))
        #print("💩💩-Avg loss eval: {:.4f} 💩💩-Accuracy eval: {:.2f}%".format(avg_loss_eval.item(), acc_eval.item()*100))

    #save model
    torch.save(model.state_dict(), "/kaggle/working/er_model(not_rename).pth")