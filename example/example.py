import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(256, 256*4)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(256*4, 10)
        
    def forward(self, x):
        return self.layer_2(self.relu(self.layer_1(x)))
    


# this is just an example function to test submitting jobs with slurm
def main():

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    inputs = torch.randn(100, 256).cuda()
    labels = torch.randint(0, 10, size=(100, )).cuda()
    print(inputs.shape)
    print(labels.shape)
    mynet = MyNet().cuda()
    optimizer = torch.optim.AdamW(mynet.parameters(), lr=0.0001)
    for _ in range(100):
        preds = mynet(inputs)
        print(preds.shape)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        print(f"epoch {_+1} loss: {loss.data}")
        optimizer.step()
        optimizer.zero_grad()
    
    with torch.no_grad():
        prediction = mynet(inputs).detach().cpu()
        prediction = prediction.argmax(dim=-1)
    
    print(f"accuracy {(prediction == labels.cpu()).sum() / len(prediction):.2f}")

    
    
    

if __name__ == "__main__":
    main()