import torch, os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader


def twomoon(num_of_data,sigma):
    """sigma control the variance of addition noise(es)"""
    rng = np.random.default_rng(seed=1)
    es1 = rng.normal(0, sigma, (num_of_data,2)) 
    es2 = rng.normal(0, sigma, (num_of_data,2))
    us = rng.random(num_of_data)*np.pi
    xs = np.zeros((num_of_data,2))
    ys = np.zeros((num_of_data,2))   
    for i in range(int(num_of_data)):
        if i % 2 == 0:  # negative examples
            xs[i,:] = [0,1]
            ys[i][0] = np.cos(us[i])+1/2+ es1[i][0]
            ys[i][1] = np.sin(us[i])-1/6+ es1[i][1]
        else:  # positive examples
            xs[i,:] = [1,0]
            ys[i][0] = np.cos(us[i])-1/2+ es2[i][0]
            ys[i][1] = -np.sin(us[i])+1/6+ es2[i][1]
    return xs,ys  # [num_of_data,2],[num_of_data,2] array float32

def simulation(num_of_data,method=1):
    """ey stdy are the true condtional mean and std"""
    rng = np.random.default_rng(seed=1)
    ys = np.zeros(num_of_data)
    ey = np.zeros(num_of_data)
    stdy = np.zeros(num_of_data)
    if method == 1: #additive error term 
        xs = rng.normal(size=(num_of_data, 5))
        ers = rng.normal(size=(num_of_data))
        for i in range(num_of_data):
            ey[i] = xs[i,0]**2+np.exp(xs[i,1]+xs[i,2]/3)+np.sin(xs[i,3]+xs[i,4])
            ys[i] = ey[i]+ers[i]
            stdy[i] = 1

    if method == 2: #multiplicative non-Gaussian error
        xs = rng.normal(size=(num_of_data, 5))
        ers = rng.normal(0,2,size=num_of_data)
        for i in range(num_of_data):
            const = 5+xs[i,0]**2/3+xs[i,1]**2+xs[i,2]**2+xs[i,3]+xs[i,4]
            ys[i] = const*np.exp(0.25*ers[i])
            ey[i] = const*np.exp(1/16)
            stdy[i] = np.abs(const)*np.sqrt(np.exp(1/4)-np.exp(1/8))

    if method == 3: # mixture of two very close normal 
        xs = rng.normal(size=(num_of_data, 2))
        #xs = np.ones_like(xs)
        ers = rng.normal(size=(num_of_data,2))
        bs = rng.binomial(1,0.3,num_of_data)
        for i in range(num_of_data):
            ey[i] = 0.4+0.4*xs[i,0]+0.2*xs[i,1]
            stdy[i] = np.sqrt(31/40+0.84*(1+xs[i,0]+0.5*xs[i,1])**2)
            ys[i] = bs[i]*(-1-xs[i,0]-0.5*xs[i,1]+0.5*ers[i,0])+(1-bs[i])*(1+xs[i,0]+0.5*xs[i,1]+1*ers[i,1])
    return xs, ys, ey, stdy # x,y, epct of y given x, std of y given x


def create_celeba_dataset():
    """Path of CelebA Dataset"""
    IMAGE_PATH = './img_align_celeba'
    CSV_PATH = './list_attr_celeba.csv'
    #df = pd.read_csv(CSV_PATH)
    #labels = df[["image_id","Male","Young","Eyeglasses","Bald","Mustache","Smiling"]].values
    class CelebaData(Dataset):
        """Custom Dataset for loading CelebA face images"""
        def __init__(self,
                     txt_path, 
                     img_dir, 
                     transform=None):
            df = pd.read_csv(txt_path)
            self.labels = df[["Male","Young","Eyeglasses","Bald","Mustache","Smiling"]].values # labels.shape = (202599,6)
            self.image_paths = [os.path.abspath(os.path.join(img_dir, p)) for p in sorted(os.listdir(img_dir))] # sorted folder.shape = (202599,2) 
            self.transform = transform
            self.transform2= transforms.Compose([transforms.CenterCrop(148),transforms.Resize(96)])
        """image loader"""
        def load_image(self, index):
            image_path = self.image_paths[index]
            real_img = Image.open(image_path)
            return real_img
        """Show the trained picture"""    
        def show_image(self,index):
            image_path = self.image_paths[index]
            real_img = Image.open(image_path)
            img = self.transform2(real_img)
            return img

        def __len__(self):
            return len(self.image_paths)

        """Return of Dataset"""
        def __getitem__(self, index):
            img = self.load_image(index)
            img = self.transform(img)
            label =  torch.tensor(self.labels[index])
            return img, label

    """define the transform of CelebA image"""
    image_size = 96
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return CelebaData(CSV_PATH,IMAGE_PATH,transform)
    # myDataset.show_image(0) to check image
    # next(iter(myDataset)) to obtain training sample

def celeba_loader(batch_size=32,shuffle=False):
    return DataLoader(dataset=create_celeba_dataset(),batch_size=batch_size,shuffle=shuffle)

def mnist_loader(batch_size = 100):
    mnist_trainset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)  
    mnist_testset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
    train_loader = torch.utils.data.DataLoader(
                 dataset=mnist_trainset,
                 batch_size=batch_size,
                 shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                dataset=mnist_testset,
                batch_size=batch_size,
                shuffle=False)
    return train_loader,test_loader

def Chop_mnist_upperleft(images): #chop the MNIST images into upper left quadrant and the rest
    #images = images.reshape([10,784])
    index=np.array([])
    rest =np.array([])
    for i in range(0,14):
        array = np.arange((0+28*i),(14+28*i))
        rest_array = np.arange((14+28*i),(28+28*i))
        index = np.append(index,array)
        rest = np.append(rest,rest_array)
    for i in range(14,28):
        rest_array = np.arange((0+28*i),(28+28*i))
        rest = np.append(rest,rest_array)
    indices = torch.tensor(index,dtype=torch.int32)
    rest_indices = torch.tensor(rest,dtype=torch.int32)
    #select the image based on the indices of upper left position
    imgs_x = torch.index_select(images,1,indices) 
    imgs_y =torch.index_select(images,1,rest_indices)
    return imgs_x,imgs_y #[batch_size,196],[batch_size,588]
def Chop_mnist_upperlefthalf(images): #chop the MNIST images into upper left quadrant and the rest
    #images = images.reshape([10,784])
    index=np.array([])
    rest =np.array([])
    for i in range(0,14):
        array = np.arange((0+28*i),(28+28*i))
        index = np.append(index,array)
    for i in range(14,28):
        array = np.arange((0+28*i),(14+28*i))
        rest_array = np.arange((14+28*i),(28+28*i))
        index = np.append(index,array)
        rest = np.append(rest,rest_array)
    indices = torch.tensor(index,dtype=torch.int32)
    rest_indices = torch.tensor(rest,dtype=torch.int32)
    #select the image based on the indices of upper left position
    imgs_x = torch.index_select(images,1,indices) 
    imgs_y =torch.index_select(images,1,rest_indices)
    return imgs_x,imgs_y #[batch_size,588],[batch_size,196]
def Chop_mnist_lefthalf(images): #chop the MNIST images into upper left quadrant and the rest
    #images = images.reshape([10,784])
    index=np.array([])
    rest =np.array([])
    for i in range(0,28):
        array = np.arange((0+28*i),(14+28*i))
        rest_array = np.arange((14+28*i),(28+28*i))
        index = np.append(index,array)
        rest = np.append(rest,rest_array)
    indices = torch.tensor(index,dtype=torch.int32)
    rest_indices = torch.tensor(rest,dtype=torch.int32)
    #select the image based on the indices of upper left position
    imgs_x = torch.index_select(images,1,indices) 
    imgs_y =torch.index_select(images,1,rest_indices)
    return imgs_x,imgs_y #[batch_size,392],[batch_size,392]
