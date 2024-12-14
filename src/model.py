import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, nz, ngf, nc):
    super(Generator, self).__init__()

    self.seq = nn.Sequential(
      nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(),
      nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(),

      nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(),
      nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(),

      nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
      nn.Tanh() 
    )

  def forward(self, input):
    x = self.seq(input)
    return x


class Discriminator(nn.Module):
  def __init__(self, nc, ndf):
    super(Discriminator, self).__init__()
    self.seq = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid() 
    )

  def forward(self, input):
    x = self.seq(input)
    return x