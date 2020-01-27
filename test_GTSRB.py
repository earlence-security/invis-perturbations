import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from GTSRBNet import GTSRBNet
from GTSRBDataset import GTSRBDataset

def main(argv):
    root = ''
    epochs = 25

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])
    
    validset = GTSRBDataset('valid_us.npz', transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=256,
                                              shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GTSRBNet()
    model.to(device)
    
    classes = []
    with open(root + 'class_semantics.txt') as f:
        for line in f:
            classes.append(line.strip())

    checkpoint = torch.load('checkpoint_us.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100.0 * correct / total
    print('Val accuracy: %.3f' % (val_acc))

if __name__ == '__main__':
    main(sys.argv)
