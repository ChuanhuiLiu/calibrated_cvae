class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(CVAE, self).__init__()
        y_dim = 784-x_dim #correct
        # encoder part
        self.fc1 = nn.Linear(x_dim+y_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim+x_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, y_dim)
        
    def encoder(self, x,y):
        h = torch.cat((x,y),dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z,x):
        h = torch.cat((z,x),dim=-1)   
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        #return F.sigmoid(self.fc6(h))
        return self.fc6(h)
    
    def forward(self, x,y):
        mu, log_var = self.encoder(x,y)
        z = self.sampling(mu, log_var)
        return self.decoder(z,x), mu, log_var
