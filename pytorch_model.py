import numpy as np
from keras.models import model_from_json

# Load the keras model and load the weights
with open("data/lp-detector/wpod-net_update1.json", 'r') as json_file:
            model_json = json_file.read()
wpod = model_from_json(model_json,custom_objects={})
wpod.load_weights('data/lp-detector/wpod-net_update1.h5')
weights=wpod.get_weights()
    
    

from torch import nn
import torch

# Class to represent the residual blocks since they are somewhat parallel
class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

#Class to represent detection block since it is parallel
class Detection(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define network layers
        self.conv2 = nn.Conv2d(128,2,3, padding=1)
        self.conv6 = nn.Conv2d(128,6,3, padding=1)
        self.sm = nn.Softmax(dim=1)

    def forward(self, inputs):
        x1 = self.sm(self.conv2(inputs))
        x2 = self.conv6(inputs)
        return torch.cat([x1,x2],dim=1)   
    
def load_wpod():
    # Define model architecture
    model = model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16,eps=0.001, momentum=0.99),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.BatchNorm2d(16,eps=0.001, momentum=0.99),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32,eps=0.001, momentum=0.99),
        nn.ReLU(),    
        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(32, 32, 3, padding=1),             #| 
                nn.BatchNorm2d(32,eps=0.001, momentum=0.99), #|
                nn.ReLU(),                                   #|
                nn.Conv2d(32, 32, 3, padding=1),             #| 
                nn.BatchNorm2d(32,eps=0.001, momentum=0.99), #|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------|
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64,eps=0.001, momentum=0.99),
        nn.ReLU(),
        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
                nn.ReLU(),                                   #|
                nn.Conv2d(64, 64, 3, padding=1),             #|
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------| 

        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
                nn.ReLU(),                                   #|
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------| 
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(64, 64, 3, padding=1),                        #In code not paper 
        nn.BatchNorm2d(64,eps=0.001, momentum=0.99),
        nn.ReLU(),                                              #In code not paper
        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
                nn.ReLU(),                                   #|
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------|

        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
                nn.ReLU(),                                   #|
                nn.Conv2d(64, 64, 3, padding=1),             #| 
                nn.BatchNorm2d(64,eps=0.001, momentum=0.99), #|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------| 
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128,eps=0.001, momentum=0.99),
        nn.ReLU(),                                              #In code not paper
        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(128, 128, 3, padding=1),           #|
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|
                nn.ReLU(),                                   #|
                nn.Conv2d(128, 128, 3, padding=1),           #|
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------|

        # ----------------------------------------------------|
        ResBlock(                                            #|
            nn.Sequential(                                   #| 
                nn.Conv2d(128, 128, 3, padding=1),           #| 
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|
                nn.ReLU(),                                   #|
                nn.Conv2d(128, 128, 3, padding=1),           #| 
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|
            )                                                #| 
        ),                                                   #|
        nn.ReLU(),                                           #| 
        # ----------------------------------------------------| 

        # ----------------------------------------------------| In code not paper |
        ResBlock(                                            #|                   |
            nn.Sequential(                                   #|                   | 
                nn.Conv2d(128, 128, 3, padding=1),           #|                   |
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|                   |
                nn.ReLU(),                                   #|                   |
                nn.Conv2d(128, 128, 3, padding=1),           #|                   |
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|                   |
            )                                                #|                   |
        ),                                                   #|                   |
        nn.ReLU(),                                           #|                   |
        # ----------------------------------------------------| In code not paper |

        # ----------------------------------------------------| In code not paper |
        ResBlock(                                            #|                   |
            nn.Sequential(                                   #|                   | 
                nn.Conv2d(128, 128, 3, padding=1),           #|                   |
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|                   |
                nn.ReLU(),                                   #|                   |
                nn.Conv2d(128, 128, 3, padding=1),           #|                   |
                nn.BatchNorm2d(128,eps=0.001, momentum=0.99),#|                   |
            )                                                #|                   |
        ),                                                   #|                   |
        nn.ReLU(),                                           #|                   |
        # ----------------------------------------------------| In code not paper | 
        Detection()
    )
    
    #Load the weights from keras into the pytorch model
    model[0].weight.data=torch.from_numpy(np.transpose(weights[0],(3, 2, 0, 1)))
    model[0].bias.data=torch.from_numpy(weights[1])
    model[1].weight.data=torch.from_numpy(weights[2])
    model[1].bias.data=torch.from_numpy(weights[3])
    model[1].running_mean.data=torch.from_numpy(weights[4])
    model[1].running_var.data=torch.from_numpy(weights[5])
    model[3].weight.data=torch.from_numpy(np.transpose(weights[6],(3, 2, 0, 1)))
    model[3].bias.data=torch.from_numpy(weights[7])
    model[4].weight.data=torch.from_numpy(weights[8])
    model[4].bias.data=torch.from_numpy(weights[9])
    model[4].running_mean.data=torch.from_numpy(weights[10])
    model[4].running_var.data=torch.from_numpy(weights[11])
    model[7].weight.data=torch.from_numpy(np.transpose(weights[12],(3, 2, 0, 1)))
    model[7].bias.data=torch.from_numpy(weights[13])
    model[8].weight.data=torch.from_numpy(weights[14])
    model[8].bias.data=torch.from_numpy(weights[15])
    model[8].running_mean.data=torch.from_numpy(weights[16])
    model[8].running_var.data=torch.from_numpy(weights[17])
    model[10].module[0].weight.data=torch.from_numpy(np.transpose(weights[18],(3, 2, 0, 1)))
    model[10].module[0].bias.data=torch.from_numpy(weights[19])
    model[10].module[1].weight.data=torch.from_numpy(weights[20])
    model[10].module[1].bias.data=torch.from_numpy(weights[21])
    model[10].module[1].running_mean.data=torch.from_numpy(weights[22])
    model[10].module[1].running_var.data=torch.from_numpy(weights[23])
    model[10].module[3].weight.data=torch.from_numpy(np.transpose(weights[24],(3, 2, 0, 1)))
    model[10].module[3].bias.data=torch.from_numpy(weights[25])
    model[10].module[4].weight.data=torch.from_numpy(weights[26])
    model[10].module[4].bias.data=torch.from_numpy(weights[27])
    model[10].module[4].running_mean.data=torch.from_numpy(weights[28])
    model[10].module[4].running_var.data=torch.from_numpy(weights[29])
    model[13].weight.data=torch.from_numpy(np.transpose(weights[30],(3, 2, 0, 1)))
    model[13].bias.data=torch.from_numpy(weights[31])
    model[14].weight.data=torch.from_numpy(weights[32])
    model[14].bias.data=torch.from_numpy(weights[33])
    model[14].running_mean.data=torch.from_numpy(weights[34])
    model[14].running_var.data=torch.from_numpy(weights[35])
    model[16].module[0].weight.data=torch.from_numpy(np.transpose(weights[36],(3, 2, 0, 1)))
    model[16].module[0].bias.data=torch.from_numpy(weights[37])
    model[16].module[1].weight.data=torch.from_numpy(weights[38])
    model[16].module[1].bias.data=torch.from_numpy(weights[39])
    model[16].module[1].running_mean.data=torch.from_numpy(weights[40])
    model[16].module[1].running_var.data=torch.from_numpy(weights[41])
    model[16].module[3].weight.data=torch.from_numpy(np.transpose(weights[42],(3, 2, 0, 1)))
    model[16].module[3].bias.data=torch.from_numpy(weights[43])
    model[16].module[4].weight.data=torch.from_numpy(weights[44])
    model[16].module[4].bias.data=torch.from_numpy(weights[45])
    model[16].module[4].running_mean.data=torch.from_numpy(weights[46])
    model[16].module[4].running_var.data=torch.from_numpy(weights[47])
    model[18].module[0].weight.data=torch.from_numpy(np.transpose(weights[48],(3, 2, 0, 1)))
    model[18].module[0].bias.data=torch.from_numpy(weights[49])
    model[18].module[1].weight.data=torch.from_numpy(weights[50])
    model[18].module[1].bias.data=torch.from_numpy(weights[51])
    model[18].module[1].running_mean.data=torch.from_numpy(weights[52])
    model[18].module[1].running_var.data=torch.from_numpy(weights[53])
    model[18].module[3].weight.data=torch.from_numpy(np.transpose(weights[54],(3, 2, 0, 1)))
    model[18].module[3].bias.data=torch.from_numpy(weights[55])
    model[18].module[4].weight.data=torch.from_numpy(weights[56])
    model[18].module[4].bias.data=torch.from_numpy(weights[57])
    model[18].module[4].running_mean.data=torch.from_numpy(weights[58])
    model[18].module[4].running_var.data=torch.from_numpy(weights[59])
    model[21].weight.data=torch.from_numpy(np.transpose(weights[60],(3, 2, 0, 1)))
    model[21].bias.data=torch.from_numpy(weights[61])
    model[22].weight.data=torch.from_numpy(weights[62])
    model[22].bias.data=torch.from_numpy(weights[63])
    model[22].running_mean.data=torch.from_numpy(weights[64])
    model[22].running_var.data=torch.from_numpy(weights[65])
    model[24].module[0].weight.data=torch.from_numpy(np.transpose(weights[66],(3, 2, 0, 1)))
    model[24].module[0].bias.data=torch.from_numpy(weights[67])
    model[24].module[1].weight.data=torch.from_numpy(weights[68])
    model[24].module[1].bias.data=torch.from_numpy(weights[69])
    model[24].module[1].running_mean.data=torch.from_numpy(weights[70])
    model[24].module[1].running_var.data=torch.from_numpy(weights[71])
    model[24].module[3].weight.data=torch.from_numpy(np.transpose(weights[72],(3, 2, 0, 1)))
    model[24].module[3].bias.data=torch.from_numpy(weights[73])
    model[24].module[4].weight.data=torch.from_numpy(weights[74])
    model[24].module[4].bias.data=torch.from_numpy(weights[75])
    model[24].module[4].running_mean.data=torch.from_numpy(weights[76])
    model[24].module[4].running_var.data=torch.from_numpy(weights[77])
    model[26].module[0].weight.data=torch.from_numpy(np.transpose(weights[78],(3, 2, 0, 1)))
    model[26].module[0].bias.data=torch.from_numpy(weights[79])
    model[26].module[1].weight.data=torch.from_numpy(weights[80])
    model[26].module[1].bias.data=torch.from_numpy(weights[81])
    model[26].module[1].running_mean.data=torch.from_numpy(weights[82])
    model[26].module[1].running_var.data=torch.from_numpy(weights[83])
    model[26].module[3].weight.data=torch.from_numpy(np.transpose(weights[84],(3, 2, 0, 1)))
    model[26].module[3].bias.data=torch.from_numpy(weights[85])
    model[26].module[4].weight.data=torch.from_numpy(weights[86])
    model[26].module[4].bias.data=torch.from_numpy(weights[87])
    model[26].module[4].running_mean.data=torch.from_numpy(weights[88])
    model[26].module[4].running_var.data=torch.from_numpy(weights[89])
    model[29].weight.data=torch.from_numpy(np.transpose(weights[90],(3, 2, 0, 1)))
    model[29].bias.data=torch.from_numpy(weights[91])
    model[30].weight.data=torch.from_numpy(weights[92])
    model[30].bias.data=torch.from_numpy(weights[93])
    model[30].running_mean.data=torch.from_numpy(weights[94])
    model[30].running_var.data=torch.from_numpy(weights[95])
    model[32].module[0].weight.data=torch.from_numpy(np.transpose(weights[96],(3, 2, 0, 1)))
    model[32].module[0].bias.data=torch.from_numpy(weights[97])
    model[32].module[1].weight.data=torch.from_numpy(weights[98])
    model[32].module[1].bias.data=torch.from_numpy(weights[99])
    model[32].module[1].running_mean.data=torch.from_numpy(weights[100])
    model[32].module[1].running_var.data=torch.from_numpy(weights[101])
    model[32].module[3].weight.data=torch.from_numpy(np.transpose(weights[102],(3, 2, 0, 1)))
    model[32].module[3].bias.data=torch.from_numpy(weights[103])
    model[32].module[4].weight.data=torch.from_numpy(weights[104])
    model[32].module[4].bias.data=torch.from_numpy(weights[105])
    model[32].module[4].running_mean.data=torch.from_numpy(weights[106])
    model[32].module[4].running_var.data=torch.from_numpy(weights[107])
    model[34].module[0].weight.data=torch.from_numpy(np.transpose(weights[108],(3, 2, 0, 1)))
    model[34].module[0].bias.data=torch.from_numpy(weights[109])
    model[34].module[1].weight.data=torch.from_numpy(weights[110])
    model[34].module[1].bias.data=torch.from_numpy(weights[111])
    model[34].module[1].running_mean.data=torch.from_numpy(weights[112])
    model[34].module[1].running_var.data=torch.from_numpy(weights[113])
    model[34].module[3].weight.data=torch.from_numpy(np.transpose(weights[114],(3, 2, 0, 1)))
    model[34].module[3].bias.data=torch.from_numpy(weights[115])
    model[34].module[4].weight.data=torch.from_numpy(weights[116])
    model[34].module[4].bias.data=torch.from_numpy(weights[117])
    model[34].module[4].running_mean.data=torch.from_numpy(weights[118])
    model[34].module[4].running_var.data=torch.from_numpy(weights[119])
    model[36].module[0].weight.data=torch.from_numpy(np.transpose(weights[120],(3, 2, 0, 1)))
    model[36].module[0].bias.data=torch.from_numpy(weights[121])
    model[36].module[1].weight.data=torch.from_numpy(weights[122])
    model[36].module[1].bias.data=torch.from_numpy(weights[123])
    model[36].module[1].running_mean.data=torch.from_numpy(weights[124])
    model[36].module[1].running_var.data=torch.from_numpy(weights[125])
    model[36].module[3].weight.data=torch.from_numpy(np.transpose(weights[126],(3, 2, 0, 1)))
    model[36].module[3].bias.data=torch.from_numpy(weights[127])
    model[36].module[4].weight.data=torch.from_numpy(weights[128])
    model[36].module[4].bias.data=torch.from_numpy(weights[129])
    model[36].module[4].running_mean.data=torch.from_numpy(weights[130])
    model[36].module[4].running_var.data=torch.from_numpy(weights[131])
    model[38].module[0].weight.data=torch.from_numpy(np.transpose(weights[132],(3, 2, 0, 1)))
    model[38].module[0].bias.data=torch.from_numpy(weights[133])
    model[38].module[1].weight.data=torch.from_numpy(weights[134])
    model[38].module[1].bias.data=torch.from_numpy(weights[135])
    model[38].module[1].running_mean.data=torch.from_numpy(weights[136])
    model[38].module[1].running_var.data=torch.from_numpy(weights[137])
    model[38].module[3].weight.data=torch.from_numpy(np.transpose(weights[138],(3, 2, 0, 1)))
    model[38].module[3].bias.data=torch.from_numpy(weights[139])
    model[38].module[4].weight.data=torch.from_numpy(weights[140])
    model[38].module[4].bias.data=torch.from_numpy(weights[141])
    model[38].module[4].running_mean.data=torch.from_numpy(weights[142])
    model[38].module[4].running_var.data=torch.from_numpy(weights[143])
    model[40].conv2.weight.data=torch.from_numpy(np.transpose(weights[144],(3, 2, 0, 1)))
    model[40].conv2.bias.data=torch.from_numpy(weights[145])
    model[40].conv6.weight.data=torch.from_numpy(np.transpose(weights[146],(3, 2, 0, 1)))
    model[40].conv6.bias.data=torch.from_numpy(weights[147])
    
    return model