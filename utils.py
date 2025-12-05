import torch

def predict_attributes(model, image_tensor, device="cpu"):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    predictions = {k: torch.argmax(v, dim=1).item() if k!="accessories" else (v>0.5).int().tolist()
                   for k,v in outputs.items()}
    return predictions
