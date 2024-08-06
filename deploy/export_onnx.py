import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import onnx
import torch

from config import parse_args
from model.resnet import Resnet

args = parse_args()

x = torch.randn(1, 3, 128, 128)

model = Resnet(args)
ckpt_path = os.path.join(os.getcwd(), 'results', 'best.pth')
ckpt_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)["model"]
model.load_state_dict(ckpt_state_dict)
model = model.eval()

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "resnet18.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
    
onnx_model = onnx.load("resnet18.onnx") 
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")
