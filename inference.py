# Model card for H-optimus-0
"""
Extract powerful features from the histology images
Mutation prediction, survival analysis, tissue classification 

Expects images of size 224x224 extracted at 0.5 MPP
10x = 1um/pixel
20x = 0.5m/pixel
"""

from huggingface_hub import login
import torch
import timm 
from torchvision import transforms 


# login to hugging face hub
login()

model = timm.create_model(    
    "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
)
model.to("cuda")
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617), 
        std=(0.211883, 0.230117, 0.177517)
    ),
])

input = torch.rand(3, 224, 224)
input = transforms.ToPILImage()(input)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    with torch.inference_mode():
        features = model(transform(input).unsqueeze(0).to("cuda"))

assert features.shape == (1, 1536)