import torch
import torch.nn as nn



class FusionModel(nn.Module):
    def __init__(self, encoder):
        super(FusionModel, self).__init__()

        self.encoder = encoder
        self.encoder.fc = nn.Identity()
# #efficientNetV2
        self.fusion = nn.Sequential(
            nn.Linear(2560, 1280),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(640, 320),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(160, 1, bias=False)
        )
# Resnet101
#         self.fusion = nn.Sequential(
#             nn.Linear(4096, 2048),
#             nn.MaxPool1d(2, 2),
#             # nn.AvgPool1d(2,2),
#             nn.ReLU(True),
#             # nn.Dropout(p=0.3),
#             nn.Linear(1024, 512),
#             nn.MaxPool1d(2, 2),
#             # nn.AvgPool1d(2, 2),
#             nn.ReLU(True),
#             # nn.Dropout(p=0.3),
#             nn.Linear(256, 1, bias=False)
#         )
#ConvNeXt_S
        # self.fusion = nn.Sequential(
        #     nn.Linear(1536, 768),
        #     nn.MaxPool1d(2, 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(384, 192),
        #     nn.MaxPool1d(2, 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(96, 1, bias=False)
        # )
# nfnet_f0
#         self.fusion = nn.Sequential(
#              nn.Linear(6144, 3072),
#              nn.MaxPool1d(2, 2),
#              # nn.AvgPool1d(2,2),
#              nn.ReLU(True),
#              nn.Dropout(p=0.3),
#              nn.Linear(1536, 768),
#              nn.MaxPool1d(2, 2),
#                     # nn.AvgPool1d(2, 2),
#              nn.ReLU(True),
#              nn.Dropout(p=0.3),
#              nn.Linear(384, 1, bias=False)
#          )
#regnet16
        # self.fusion = nn.Sequential(
        #      nn.Linear(6048, 3024),
        #      nn.MaxPool1d(2, 2),
        #      # nn.AvgPool1d(2,2),
        #      nn.ReLU(True),
        #      nn.Dropout(p=0.3),
        #      nn.Linear(1512, 756),
        #      nn.MaxPool1d(2, 2),
        #             # nn.AvgPool1d(2, 2),
        #      nn.ReLU(True),
        #      nn.Dropout(p=0.3),
        #      nn.Linear(378, 1, bias=False)
        #  )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x_1, x_2):
        with torch.no_grad():
            x_1 = self.encoder(x_1)
            x_2 = self.encoder(x_2)

        x = torch.cat((x_1, x_2), 1)
        x = x.view(x.size(0), 1, -1)
        x = self.fusion(x)
        x = x.squeeze()
        return x



