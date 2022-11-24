from sklearn import datasets
from matplotlib import pyplot as plt 
from glow import * 
from utils import * 

if __name__  == '__main__':
    
    n_samples = 500
    data, _ = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=0
    )
   