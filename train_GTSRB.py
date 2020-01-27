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
    
    trainset = GTSRBDataset('train_us.npz', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)
    validset = GTSRBDataset('valid_us.npz', transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=256,
                                              shuffle=True, num_workers=2)
    testset = GTSRBDataset('test_us.npz', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GTSRBNet()
    model.to(device)
    
    lossFn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

    classes = []
    with open(root + 'class_semantics.txt') as f:
        for line in f:
            classes.append(line.strip())

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = lossFn(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                                  (epoch + 1, i + 1, run_loss / 2000))
                run_loss = 0.0

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in validloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.long()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        print('[%d] accuracy: %.3f' % (epoch + 1, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, 'checkpoint_us.tar')
    
if __name__ == '__main__':
    main(sys.argv)
