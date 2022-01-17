import pandas as pd
from collections import OrderedDict
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from models import ResNet18
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (
    torch.cuda.is_available()) else "cpu")


def unpack_dict(PATH):

    network = ResNet18()
    init = torch.load(PATH,  map_location=device)
    new_state_dict = OrderedDict()
    for k, v in init.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    network.load_state_dict(new_state_dict)
    without_BN_layers = []
    for name, weight in network.named_parameters():
        #        if not (name.find('bn') != -1):
        nump = np.array(weight.cpu().detach().numpy().flatten())
#            if not (np.all(nump == nump[0])):
        without_BN_layers.extend(weight.cpu().detach().numpy().flatten())
    return np.array(without_BN_layers)


# model_PATH = path+'final_trained_network_q='+str(q)
n = 1
directory = 'iteration-'+str(n)
path = '/final_trained_network_q='

# directory index

init = np.abs(unpack_dict(directory+'/Initialization-'+str(n)))

plt.hist(init, bins=2000, range=[0, 0.05],
         rwidth=0.5, alpha=0.5, label='SMD-L1', color='b')
plt.xlim(xmin=0)
plt.title("Absolute value of Weights Initialization")
plt.legend()
plt.savefig('initialization.pdf')
plt.close()


for i in [1, 2, 3, 10]:
    weights = np.abs(unpack_dict(directory+path+str(i)))

    plt.hist(weights, bins=2000, rwidth=0.5, label='SMD-L'+str(i), color='b')
    plt.xlim(xmin=0)
    plt.title("Absolute value of Weights SMD with L"+str(i))
    plt.legend()
    plt.savefig(directory + '/weights_'+str(i) + '.pdf')
    plt.close()
