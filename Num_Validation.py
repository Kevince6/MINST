import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from Num_Model import CNN

device = torch.device("cuda")

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,),(0.3081))])
test_set = datasets.MNIST("MINST",train=False,transform=trans,download=True)
test_loader = DataLoader(test_set,batch_size=64,shuffle=False)

test_set_len = len(test_set)

model = CNN()
model = model.to(device)
model.load_state_dict(torch.load("NumRecog_method1.pth"))

loss = nn.CrossEntropyLoss()
loss = loss.to(device)

correct = 0
test_loss = 0.0

for data in test_loader:
    imgs,labels = data
    imgs = imgs.to(device)
    labels = labels.to(device)
    output = model(imgs)
    test_l = loss(output,labels)
    test_loss += test_l.item()
    correct += (output.argmax(1)==labels).sum().item()
    
accuracy = correct/test_set_len

print(f"Average Loss: {test_loss/test_set_len}")
print(f"Accuracy: {accuracy}")
