import torch
from torch import nn
from torch.nn import Sequential,Linear,Conv2d,MaxPool2d,Flatten,ReLU,BatchNorm2d,BatchNorm1d,Dropout

class CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Sequential(
            Conv2d(1,32,5,1,2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),  #32*14*14
            
            Conv2d(32,32,5,1,2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),  #32*7*7
            
            Conv2d(32,64,3,1,1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2,ceil_mode=True), #64*4*4
            
            Flatten(),
            Linear(64*4*4,64),
            BatchNorm1d(64),
            ReLU(),
            Dropout(0.3),
            Linear(64,10)        
        )
        
    def forward(self,input):
        output = self.model(input)
        return output
        
if __name__ == "__main__":
    model = CNN()
    input = torch.randn([64,1,28,28])
    output = model(input)
    print(output.shape)
    