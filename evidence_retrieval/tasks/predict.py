import torch
import pandas as pd
from tqdm import tqdm

from text_module.text_embedding import TextEmbedding
from load_data.data_loader import Dataset, getDataloader
from models.model import PickEvidenceIndexModel

def predict(model, text_embedding, test_loader, device):
    model.eval()
    prediction = {"id": [], "evidence": []}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="ĐANG TIẾN HÀNH LẤY KẾT QUẢ DỰ ĐOÁN"):
            id, context, claim = batch["id"], batch["context"], batch["claim"]
            
            #embedding
            input_ids = text_embedding(claim, context).to(device)

            #forward call
            out, out_soft = model(input_ids)
            
            #append res
            prediction["id"] = prediction["id"] + id
            prediction["evidence"] = prediction["evidence"] + out_soft.argmax(dim=-1).detach().numpy().tolist()

    return prediction

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("❤️❤️❤️❤️❤️ DEVICE LÀ", device)

    text_embedding = TextEmbedding(config).to(device)
    print("❤️❤️❤️❤️❤️ KHỞI TẠO EMBEDDING XONG.")

    test_dataset = Dataset(config["data_path"]["test"])
    test_loader = getDataloader(Dataset, config["predict"]["batch_size"])
    print("❤️❤️❤️❤️❤️ LOAD DATA THÀNH CÔNG.")

    model = PickEvidenceIndexModel(config).to(device)
    print("❤️❤️❤️❤️❤️ LOAD MODEL THÀNH CÔNG.")

    model.load_state_dict(torch.load(config["checkpoint"]))
    print("❤️❤️❤️❤️❤️ LOAD MODEL ĐÃ FINE TUNE THÀNH CÔNG.")

    prediction = predict(model, text_embedding, test_loader, device)

    pd.DataFrame(prediction).to_csv(config["result_path"])



