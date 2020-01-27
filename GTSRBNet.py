import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GTSRBNet(nn.Module):
    def __init__(self):
        super(GTSRBNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 3, 1)

        self.conv2_1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2_1 = nn.MaxPool2d(2)
        self.drop2_1 = nn.Dropout2d(0.5)

        self.conv3_1 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool3_1 = nn.MaxPool2d(2)
        self.drop3_1 = nn.Dropout2d(0.5)

        self.conv4_1 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv4_2 = nn.Conv2d(128, 128, 5, padding=2)
        self.pool4_1 = nn.MaxPool2d(2)
        self.drop4_1 = nn.Dropout2d(0.5)

        # flatten

        self.fc5_1 = nn.Linear(4*4*128, 1024)
        self.drop5_1 = nn.Dropout(0.5)
        self.fc6_1 = nn.Linear(1024, 1024)
        self.drop6_1 = nn.Dropout(0.5)
        self.fc7_1 = nn.Linear(1024, 43)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))

        x = F.relu(self.conv2_1(x))
        x2 = self.drop2_1(self.pool2_1(F.relu(self.conv2_2(x))))

        x = F.relu(self.conv3_1(x2))
        x3 = self.drop3_1(self.pool3_1(F.relu(self.conv3_2(x))))

        x = F.relu(self.conv4_1(x3))
        x4 = self.drop4_1(self.pool4_1(F.relu(self.conv4_2(x))))

        x = x4.view(-1, 4*4*128)

        x = self.drop5_1(F.relu(self.fc5_1(x)))
        x = self.drop6_1(F.relu(self.fc6_1(x)))
        x = self.fc7_1(x)
        return x

    def predict(self, image):
        assert(torch.max(image) <= 0.5 and torch.min(image) >= -0.5)
        self.eval()
        with torch.no_grad():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            image_batch = torch.zeros((1, 3, 32, 32))
            image_batch[0, :, :, :] = image
            img = image_batch.to(device)
            output = self(img)
            _, predict = torch.max(output.data, 1)
        return predict[0].item()
