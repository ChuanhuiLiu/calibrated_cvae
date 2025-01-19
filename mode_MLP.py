import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(seed=1)
""" MLP q(z|x, y) """
class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size,activation):  # layer_sizes=[px, h1, ..., hl]
        super().__init__()
        if activation == "ReLU":
            act_func =  nn.ReLU()
        if activation == "Tanh":
            act_func =  nn.ReLU()
        if activation == "Sigmoid":
            act_func =  nn.ReLU()
        self.MLP = nn.Sequential()  # [px+py, h1, ..., hl]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            
        self.linear_mean = nn.Linear(layer_sizes[-1], latent_size)  # [hl, pz]
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, y=None):
        x = torch.cat((x, y), dim=-1)
        x = self.MLP(x)
        mean = self.linear_mean(x)
        log_var = self.linear_log_var(x)
        return mean, log_var  # [batch_size, pz]

""" MLP mu(y|x,z) """
class Decoder(nn.Module):
    def __init__(self, layer_sizes, input_size,activation):  # layer_sizes=[hl, ..., h1, (px+py)*nrep]
        super().__init__()
        self.MLP = nn.Sequential()  #
        input_size = input_size
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            #print("in_size=",in_size) 2
            #print("out_size=",out_size) 4
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            self.linear = nn.Linear(out_size, out_size)
    def forward(self, z,x):
        z = torch.cat((z, x), dim=-1)
        u = self.MLP(z)
        u = self.linear(u)
        return u

class CVAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, px, py, decoder_layer_sizes):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.latent_size = latent_size
        self.px = px
        self.py = py
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes,input_size=latent_size+px)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    
    def forward(self, x, y):
        mean, log_var = self.encoder(x, y)  # [batch_size, latent_size]
        z = self.reparameterize(mean, log_var)  # [batch_size, nrep] 
        pred_y_mean = self.decoder(z,x) #+ noise
        return pred_y_mean , mean, log_var
    
    def validate(self,pred_y_mean,x,gamma):
        pred_y_mean = pred_y_mean.detach().cpu().numpy()
        noise = gamma* rng.normal(0, 1, (len(x),2)) 
        pred_y = pred_y_mean + noise
        x= x.detach().cpu().numpy()
        plt.scatter(pred_y[:,0],pred_y[:,1],c=x)
        plt.show()
        return 
    def test(self,gamma):
        n = 5000
        origin_x = (np.arange(n)>n/2)
        test_x1 = origin_x.reshape(n,1)
        test_x2 = 1 - test_x1
        test_x = np.concatenate((test_x1, test_x2), axis=1)# [2350,]
        z = rng.normal(0,1,(n,2))  
        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        z = torch.from_numpy(z).type(torch.FloatTensor)
        test_x = test_x.to(device)
        z = z.to(device)
        sampled_y_mean = self.decoder(z,test_x).detach().cpu().numpy()
        noise = gamma* rng.normal(0,1,(n,2))  
        sampled_y = sampled_y_mean + noise
        plt.scatter(sampled_y[:,0],sampled_y[:,1],c=origin_x)
        plt.show()
        return 
        return

criterion = nn.GaussianNLLLoss(reduction='sum')
criterion2 = nn.MSELoss(reduction="sum")

# Lower bound for maximum likelihood
def loss_fn(recon_y, y, mean, log_var,gamma):
    var = gamma**2 * torch.ones(len(y),device = y.device)
    NLK = criterion(recon_y, y,var)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    SSE = criterion2(recon_y,y)/y.shape[1] #
    return [NLK + KLD, SSE,KLD]