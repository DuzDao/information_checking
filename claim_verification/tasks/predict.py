import torch
from tqdm import tqdm

def eval(model, text_embedding, eval_loader, criterion, device, epoch):
    model.eval()
    avg_loss = 0
    total_true_pred = 0
    total_sample = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"EVALUATING EPOCH {epoch}"):
            claim, evidence, label = batch["claim"], batch["evidence"], batch["label"]

            #text_embedding
            input_ids = text_embedding(claim, evidence).to(device)

            #forward call
            out, out_soft = model(input_ids)
            loss = criterion(out.to(device), label.to(device))
            avg_loss += loss

            #calc acc
            total_true_pred += (out_soft.argmax(dim=-1).to(device) == label.to(device)).sum()
            total_sample += label.size()[0]
    return avg_loss/len(eval_loader), total_true_pred/total_sample