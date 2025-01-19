import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(seed=1)
class CVAE(nn.Module):
    def __init__(self, in_channels = 3,
                 y_dim = 6,
                 z_dim = 32,
                 hidden_dims = 10,
                 img_size = 96,):

        super(CVAE, self).__init__()
        self.z_dim = z_dim #latent dimension
        self.y_dim = y_dim #label dimension
        self.img_size = img_size 
        
        """encoder"""
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        in_channels += 6 # To account for the extra channel from label
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(negative_slope=0.02))
            )
            in_channels = h_dim



        self.encoder_net = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1]*9, z_dim) #dimension of final layer is 2x2 if starts with 64x64
        self.var = nn.Linear(hidden_dims[-1]*9, z_dim)
        
        """decoder"""
        modules = []

        self.decoder_input = nn.Linear(z_dim + y_dim, hidden_dims[-1] * 9)

        reverse_dims = list(reversed(hidden_dims))

        for i in range(len(reverse_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reverse_dims[i],
                                       reverse_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(reverse_dims[i + 1]),
                    nn.LeakyReLU(negative_slope=0.02))
            )

        self.decoder_net = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(reverse_dims[-1],
                                               reverse_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(reverse_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(reverse_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())        
        
    def encoder(self, x,y):
        #Encodes the input by passing through the encoder network and returns the latent codes.
        #:param input: (Tensor) Input tensor "x" [N x C x H x W]
        #:param input: (Tensor) Input tensor "y" [N x Y]
        #:return: (Tensor) List of latent codes
        #embedded_y = self.embed_y(y) #[N x Y] > [N x (H x W)]
        #embedded_y = embedded_y.view(-1, self.img_size, self.img_size).unsqueeze(1) # [N x (H x W)] > [N x 1 x H x W]
        embedded_y = y.unsqueeze(-1).repeat(1, 1, self.img_size*self.img_size).reshape(y.shape[0],y.shape[1],self.img_size,self.img_size)
        embedded_x = self.embed_data(x)
        result = self.encoder_net(torch.cat([embedded_x, embedded_y], dim = 1))
        result = torch.flatten(result, start_dim=1)
        mu = self.mu(result)
        log_var = self.var(result)
        return mu, log_var # mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z,y):
        """ latent + label -> one channel -> 3 channel  """
        result = self.decoder_input(torch.cat([z, y], dim = 1))
        result = result.view(-1, 512, 3, 3)
        result = self.decoder_net(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, x,y):
        mu, log_var = self.encoder(x,y)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z,y), mu, log_var
    
