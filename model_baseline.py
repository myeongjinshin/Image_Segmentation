import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn =  nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dilations = [1, 3, 6, 12]

        self.aspp1 = _ASPPModule(in_channels,128, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(in_channels, 128, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(in_channels, 128, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(in_channels, 128, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, 128, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(128),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(640, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.1)]

        self.up = nn.Sequential(*layers)
        self.up.apply(init_weight)

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.1)]
        layers += [nn.MaxPool2d(2, stride=2, dilation=dilation, ceil_mode=True)]

        self.down = nn.Sequential(*layers)
        self.down.apply(init_weight)

    def forward(self, x):
        return self.down(x)



class UNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        n = 64
        self.dec1 = UNetDec(3, n, dropout=True)
        self.dec2 = UNetDec(n, 2*n, dropout=True)
        self.dec3 = UNetDec(2*n, 4*n, dropout=True)
        self.dec4 = UNetDec(4*n, 8*n, dropout=True, dilation=2)
        self.center = ASPP(8*n, 4*n)
        '''
        self.enc4 = UNetEnc(16*n, 8*n, 4*n, dropout=True)
        self.enc3 = UNetEnc(8*n, 4*n, 2*n, dropout=True)
        self.enc2 = UNetEnc(4*n, 2*n, n, dropout=True)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*n, n, 3),
            nn.BatchNorm2d(n),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n, n, 3),
            nn.BatchNorm2d(n),
            nn.Sigmoid(),
            nn.Dropout(.5),
        )
        '''
        self.conv1 = nn.Conv2d(2*n, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
    

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        '''
        enc4 = self.enc4(torch.cat([
            center, F.interpolate(dec4, center.size()[2:])], 1))
        enc3 = self.enc3(torch.cat([
            enc4, F.interpolate(dec3, enc4.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.interpolate(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.interpolate(dec1, enc2.size()[2:])], 1))

        return F.interpolate(self.final(enc1), x.size()[2:])
        '''
        low_level = self.conv1(dec2)
        low_level = self.bn1(low_level)
        low_level = self.relu(low_level)

        final = self.last_conv(torch.cat([low_level, F.interpolate(center, low_level.size()[2:])],1))
        return F.interpolate(final, x.size()[2:])

class MultiUNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n = 48
        self.imsz = 192
        self.dec11 = UNetDec(3, n, dropout=True)
        self.dec12 = UNetDec(n, 2*n, dropout=True)
        self.dec13 = UNetDec(2*n, 4*n, dropout=True)
        self.dec21 = UNetDec(3, n, 3, dropout=True)
        self.dec22 = UNetDec(n, 2*n, 3, dropout=True)
        self.dec23 = UNetDec(2*n, 4*n, 3, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(2*4*n, 2*8*n, 3),
            nn.BatchNorm2d(2*8*n),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2*8*n, 2*8*n, 3),
            nn.BatchNorm2d(2*8*n),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(2*8*n, 2*4*n, 2, stride=2),
            nn.BatchNorm2d(2*4*n),
            nn.LeakyReLU(inplace=True),
        )
        self.enc3 = UNetEnc(2*8*n, 2*4*n, 2*2*n, dropout=True)
        self.enc2 = UNetEnc(2*4*n, 2*2*n, 2*n, dropout=True)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*2*n, 2*n, 3),
            nn.BatchNorm2d(2*n),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2*n, 2*n, 3),
            nn.BatchNorm2d(2*n),
            nn.Sigmoid(),
            nn.Dropout(.5),
        )
        self.final = nn.Conv2d(2*n, num_classes, 1)

    def forward(self, x):
        dec11 = self.dec11(x)
        dec12 = self.dec12(dec11)
        dec13 = self.dec13(dec12)
        dec21 = self.dec21(x)
        dec22 = self.dec22(dec21)
        dec23 = self.dec23(dec22)
        '''
        center = self.center(torch.cat([dec13,dec23], 1)) 
        enc3 = self.enc3(torch.cat([center, dec13, dec23]))
        enc2 = self.enc2(torch.cat([enc3, dec12, dec22]))
        enc1 = self.enc1(torch.cat([enc2, dec11, dec21]))
        '''

        center = self.center(torch.cat([dec13,F.interpolate(dec23, dec13.size()[2:])], 1)) 
        enc3 = self.enc3(torch.cat([
            center, F.interpolate(dec13, center.size()[2:]), F.interpolate(dec23, center.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.interpolate(dec12, enc3.size()[2:]), F.interpolate(dec22, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.interpolate(dec11, enc2.size()[2:]), F.interpolate(dec21, enc2.size()[2:])], 1))

        return F.interpolate(self.final(enc1), x.size()[2:])
        



if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = segnet()
    output = model(batch)
    print(output.size())