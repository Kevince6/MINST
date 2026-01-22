import torch
from torch import nn
from torch.nn import Sequential,Linear,Flatten,Conv2d,MaxPool2d,ReLU
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Num_Model import CNN
from torch.optim import SGD
import time

device = torch.device("cuda")

trans = transforms.Compose([transforms.RandomRotation(20),
                            transforms.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.9,1.1),shear=5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,),(0.3081))
                            ])
train_set = datasets.MNIST("MINST",train=True,transform=trans,download=True)
test_set = datasets.MNIST("MINST",train=False,transform=trans,download=True)

TrainLoader = DataLoader(train_set,batch_size=64,shuffle=True,drop_last=True)
TestLoader = DataLoader(test_set,batch_size=64,shuffle=True) 

train_set_size = len(train_set)
test_set_size = len(test_set)

writer = SummaryWriter("NumRec_log")
model = CNN()
model = model.to(device)
loss = nn.CrossEntropyLoss()
loss = loss.to(device)
optimizer = SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)

epoch = 30
step = 0
start_time = time.time()
model.train()

for i in range(epoch):
    epoch_loss = 0
    for data in TrainLoader:
        imgs,labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)
        batch_loss = loss(output,labels)
        epoch_loss += batch_loss.item()
        writer.add_scalar("Batch_Loss",batch_loss.item(),step)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        step+=1
    print(f"Epoch {i+1}, Epoch loss:{epoch_loss:.4f}") 
    writer.add_scalar("Epoch Loss",epoch_loss,i+1)
 
end_time = time.time()
lapse = end_time-start_time
print(f"Training Time: {lapse:.4f}")   

correct = 0
test_loss = 0.0
model.eval()
with torch.no_grad():
    for data in TestLoader:
        imgs,labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)
        test_loss += loss(output,labels)
        correct += (output.argmax(1)==labels).sum().item()
        
accuracy = correct/test_set_size
print(f"Test loss:{test_loss:.4f}")
print(f"Accuracy:{accuracy}")
        
torch.save(model.state_dict(),"NumRecog_method1.pth")
writer.close()


    