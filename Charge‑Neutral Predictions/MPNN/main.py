1# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 22, 2019
# // PURPOSE     : main function. a wrapper of charge perdiction system for testing and evaluation
# // SPECIAL NOTES: Uses charge_prediction_system that uses data_handling.py and model.py
# // ===============================
# // Change History: 1.0: initial code: wrote and tested.
# // Change History: 2.0: updated code: added mini batches
# //
# //==================================
__author__ = "Ali Raza"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = ""
__version__ = "2.0"
__maintainer__ = "ali raza"
__email__ = "razaa@oregonstate.edu"
__status__ = "done"

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from data_handling import *
from model import *
from charge_prediction_system import *
from torch_geometric.loader import DataLoader
import numpy as np
import random
import copy
import csv
import torch
import math
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal

likelihood_num = 0
print("----------------------------------------------")
print(">>> loading parameters")

GRAPHS_LOCATION = "input"
ONE_HOT_ENCODING_CSV = "../atom_to_int.csv"
TRAINING_SET_CUT = 70  # percentage
VALIDATION_SET_CUT = 10  # percentage

MAX_EPOCHS = 3000
MAX_ITERATIONS = 10

# Baseline
BATCH_SIZE = 32
GNN_LAYERS = 4
EMBEDDING_SIZE = 10
HIDDEN_FEATURES_SIZE = 30
PATIENCE_THRESHOLD = 100
LEARNING_RATE = 0.005

# Likelihood Loss
# BATCH_SIZE = 128
# GNN_LAYERS = 5
# EMBEDDING_SIZE = 30
# HIDDEN_FEATURES_SIZE = 50
# PATIENCE_THRESHOLD = 300
# LEARNING_RATE = 0.005

# Expected Loss
# BATCH_SIZE = 128
# GNN_LAYERS = 6
# EMBEDDING_SIZE = 20
# HIDDEN_FEATURES_SIZE = 50
# PATIENCE_THRESHOLD = 300
# LEARNING_RATE = 0.005

# print("Expected Loss")
print("Baseline")
# print(f"Tuning PATIENCE_THRESHOLD: {PATIENCE_THRESHOLD}")

device = torch.device('cuda:3')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

crit = torch.nn.L1Loss()

if not (os.path.exists("results/")):
    os.mkdir('results/')
if not (os.path.exists('results/graphs')):
    os.mkdir('results/graphs')
if not (os.path.exists('results/embedding')):
    os.mkdir('results/embedding')

print("...done")
print("----------------------------------------------")

print("----------------------------------------------")
print(">>> reading graphs and generating data_list")
data_list = data_handling(GRAPHS_LOCATION, READ_LABELS = True)
print("...done")
print("----------------------------------------------")
print()
NUM_NODE_FEATURES = data_list[0]['x'].shape[1]

# dividing data into testing and training
print(">>> reading one-hot encoding")
with open(ONE_HOT_ENCODING_CSV) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    element_types = []
    one_hot_encoding = []
    next(readCSV)
    for row in readCSV:
        element_types.append(row[0])
        one_hot_encoding.append(int(row[1]))

    # sorting them
    indices_sorted_elements = np.argsort(one_hot_encoding)
    element_types = np.array(element_types)[indices_sorted_elements]
    one_hot_encoding = np.array(one_hot_encoding)[indices_sorted_elements]
print("...done")
print("-----------------------------------------------------------------------------------------")

print("----------------------------------------------")
print(">>> shuffling data for different training, validation, and testing sets each run")
data_size = len(data_list)
cut_training = int(data_size * (TRAINING_SET_CUT / 100))
cut_validation = int(data_size * (TRAINING_SET_CUT + VALIDATION_SET_CUT) / 100)
# iteration = 0
data_list_shuffled = copy.deepcopy(data_list)
# making sure training_dataset has all the elements after shuffling

dataa = data_list
loader = DataLoader(dataa, batch_size=len(dataa))
print("total MOFs: {}".format(len(dataa)))

for data in loader:
    data = data.to(device)
    label = data.y.to(device)
    features = data.x.to(device)
    print("total nodes: {}".format(len(label)))

    elements_number = len(features[0])
    total_instances_all = np.zeros(elements_number)
    total_instances_mof_all = np.zeros(elements_number)

    for element_index in range(elements_number):
        indices = (features[:, element_index] == 1)
        label_element = label[indices].cpu().numpy()
        total_instances_all[element_index] = len((label[indices]))  # number of atoms in datasets
        total_instances_mof_all[element_index] = len(
            set(data.batch[indices].cpu().numpy()))  # number of mofs containing that element

    # indices of sorted element
    indices_sorted_elements = np.argsort(total_instances_all)
    indices_sorted_elements = np.flipud(indices_sorted_elements)
    # %-----------------------------------------------------------------------

loss_all = np.zeros(MAX_ITERATIONS)
charge_sum_all = np.zeros(MAX_ITERATIONS)
mad_all = np.zeros(MAX_ITERATIONS)
print("Total MOFs: {}".format(len(dataa)))
# module for evaluating

def multivariate_mean_variance(means, sigmas):
    n = len(sigmas)

    A = torch.diag(sigmas[:-1])
    B = torch.ones(n-1, n-1).double().to(device) * torch.pow(sigmas[-1], -1)

    covariance_matrix = A - 1/(1 + torch.trace(torch.matmul(B, A))) * torch.matmul(A ,torch.matmul(B, A))

    c = (k - means[-1])/sigmas[-1]
    reduced_mean = torch.matmul(covariance_matrix, torch.ones(n-1).to(device)*c + torch.div(means[:-1], sigmas[:-1]))

    return reduced_mean, covariance_matrix

def nloglikelihood(mu, sigma, y):
    dist = MultivariateNormal(mu, sigma)
    return -dist.log_prob(y)



print("BATCH_SIZE = ", BATCH_SIZE)
print("GNN_LAYERS = ", GNN_LAYERS)
print("HIDDEN_FEATURES_SIZE =", HIDDEN_FEATURES_SIZE)
print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
print("PATIENCE_THRESHOLD = ", PATIENCE_THRESHOLD)
print("LEARNING_RATE = ", LEARNING_RATE)

train_losses = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : []
}
valid_losses = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : []
}
test_losses = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : []
}
final_loss = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : [],
    'gaussian_with_erf_loss_model_1': []
}

ll_probs = {
    'gaussian_cor' : [],
    'gaussian_cor_model_1' : [],
    'gaussian_with_erf_loss' : [],
    'gaussian_with_erf_loss_model_1': []
}

print("Method \t\t\t MAD \t\t\t Negative Log-likehood Probability")

for iteration in range(MAX_ITERATIONS):
    unique_flag = False
    while unique_flag == False:
        unique_flag = True
        random.shuffle(data_list_shuffled)
        train_dataset = data_list_shuffled[:cut_training]
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        for data in train_loader:
            data = data.to(device)
            label = data.y.to(device)
            features = data.x.to(device)
            elements_number = len(features[0])
            for element_index in range(elements_number):
                indices = (features[:, element_index] == 1)
                if len((label[indices])) == 0:  # number of atoms in datasets
                    unique_flag = False
                    break

    valid_dataset = data_list_shuffled[cut_training:cut_validation]
    test_dataset = data_list_shuffled[cut_validation:]
    train_data_size = len(train_dataset)
    valid_data_size = len(valid_dataset)
    test_data_size = len(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # systems = ['gaussian_with_erf_loss']
    # systems = ['likelihood']
    systems = ['gaussian_cor']

    for system in systems:
        if system == 'gaussian_cor':
            model1, train_loss, valid_loss, test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit, device = device)
            model2, train_loss, valid_loss, test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit, device = device)

            # loss_data = {"train_loss": train_loss, "valid_loss": valid_loss,
            #             "test_loss": test_loss}
                                
            # if not os.path.exists("LossDataBaseline"):
            #     os.makedirs("LossDataBaseline")
                
            # with open(f"LossDataBaseline/Data{iteration}.pkl", "wb") as f:
            #     pickle.dump(loss_data, f)

        # elif system == 'gaussian_with_erf_loss':
        else:
            m = len(train_loader)
            model1, model1_train_loss, model1_valid_loss, model1_test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit, device = device)
            model2, model2_train_loss, model2_valid_loss, model2_test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit, device = device)

            # loss_data = {"model1_train_loss": model1_train_loss, "model1_valid_loss": model1_valid_loss,
            #             "model1_test_loss": model1_test_loss, "model2_train_loss": model2_train_loss,
            #             "model2_valid_loss": model2_valid_loss, "model2_test_loss": model2_test_loss,}
                                
            # if not os.path.exists(f"LossData{PATIENCE_THRESHOLD}"):
            #     os.makedirs(f"LossData{PATIENCE_THRESHOLD}")
                
            # with open(f"LossData{PATIENCE_THRESHOLD}/Data{iteration}.pkl", "wb") as f:
            #     pickle.dump(loss_data, f)

            # model1 = torch.load('models.pt')[0]
            # model2 = torch.load('models.pt')[1]
        # train_losses[system].append(train_loss)
        # valid_losses[system].append(valid_loss)
        # test_losses[system].append(test_loss)
    
    # torch.save([model1, model2], 'models.pt')
    # torch.save(model1, 'model_' + system + '.pt')
    # torch.save(test_dataset, 'test_dataset.pt')
    # models = torch.load('models.pt')

    dataa = test_dataset
    # dataa = valid_dataset
    loader = DataLoader(dataa, batch_size=len(dataa))

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            label = data.y.to(device)
            features = data.x.to(device)

            for index, system in enumerate(systems):
                if system == 'gaussian_cor':
                    model1.eval()
                    model2.eval()
                    pred1, _, sigma1, uncorrected_mu1 = model1(data)
                    pred2, _, sigma2, uncorrected_mu2 = model2(data)

                    pred = (pred1 + pred2) / 2

                    llp = 0
                    llp_model1 = 0
                    llp_model2 = 0
                    for i in range(0, data.num_graphs):
                        _, sigma_matrix1 = multivariate_mean_variance(uncorrected_mu1[data.batch == i], sigma1[data.batch == i])
                        llp_model1 += nloglikelihood(pred1[data.batch == i][:-1], sigma_matrix1, label[data.batch == i][:-1])

                        _, sigma_matrix2 = multivariate_mean_variance(uncorrected_mu2[data.batch == i], sigma2[data.batch == i])
                        llp_model2 += nloglikelihood(pred2[data.batch == i][:-1], sigma_matrix2, label[data.batch == i][:-1])

                        sigma_matrix = (sigma_matrix1 + sigma_matrix2)/4
                        llp += nloglikelihood(pred[data.batch == i][:-1], sigma_matrix, label[data.batch == i][:-1])

                        # llp_max = max(llp_max, temp/len(sigma[data.batch == i]))
                        # llp_min = min(llp_min, temp/len(sigma[data.batch == i]))

                        # likelihood_data = {"mu": pred[data.batch == i], "sigma": sigma_matrix,
                        # "likelihood": temp, "ground truth": label[data.batch == i]}

                        # if not os.path.exists("LikelihoodDataBaseline"):
                        #     os.makedirs("LikelihoodDataBaseline")
                        
                        # with open(f"LikelihoodDataBaseline/Data{likelihood_num}.pkl", "wb") as f:
                        #     pickle.dump(likelihood_data, f)
                            
                        # likelihood_num += 1
                        
                    llp = llp/len(data)
                    llp_model1 = llp_model1/len(data)
                    llp_model2 = llp_model2/len(data)

                elif system == 'gaussian_cor_with_sampling':
                    pred = model1(data, False)
                else:
                    model1.eval()
                    model2.eval()

                    mu1, sigma1, _, _ = model1(data)
                    pred1 = torch.empty_like(mu1)

                    mu2, sigma2, _, _ = model2(data)
                    pred2 = torch.empty_like(mu2)
                    
                    llp = 0
                    llp_model1 = 0
                    llp_model2 = 0
                    for i in range(0, data.num_graphs):
                        mu_sample1 = mu1[data.batch == i]
                        sigma_sample1 = sigma1[data.batch == i]

                        reduced_mean1, covariance_matrix1 = multivariate_mean_variance(mu_sample1, sigma_sample1)
                        pred1[data.batch == i] = torch.cat((reduced_mean1, torch.tensor([0 - torch.sum(reduced_mean1)]).to(device)), dim=0)
                        llp_model1 += nloglikelihood(reduced_mean1, covariance_matrix1, label[data.batch == i][:-1])

                        mu_sample2 = mu2[data.batch == i]
                        sigma_sample2 = sigma2[data.batch == i]

                        reduced_mean2, covariance_matrix2 = multivariate_mean_variance(mu_sample2, sigma_sample2)
                        pred2[data.batch == i] = torch.cat((reduced_mean2, torch.tensor([0 - torch.sum(reduced_mean2)]).to(device)), dim=0)
                        llp_model2 += nloglikelihood(reduced_mean2, covariance_matrix2, label[data.batch == i][:-1])

                        reduced_mean = (reduced_mean1 + reduced_mean2)/2
                        covariance_matrix = (covariance_matrix1 + covariance_matrix2)/4
                        temp = nloglikelihood(reduced_mean, covariance_matrix, label[data.batch == i][:-1])
                        llp += temp

                        # likelihood_data = {"model1_mu": mu_sample1, "model1_sigma": sigma_sample1,
                        #                     "model2_mu": mu_sample2, "model2_sigma": sigma_sample2,
                        #                     "model1_constrained_mu": reduced_mean1, "model1_constrained_sigma": covariance_matrix1,
                        #                     "model2_constrained_mu": reduced_mean2, "model2_constrained_sigma": covariance_matrix2,
                        #                     "constrained_mu": reduced_mean, "constrained_sigma": covariance_matrix, "likelihood": temp,
                        #                     "ground truth": label[data.batch == i]}
                        # if not os.path.exists("LikelihoodData"):
                        #     os.makedirs("LikelihoodData")
                        
                        # with open(f"LikelihoodData/Data{likelihood_num}.pkl", "wb") as f:
                        #     pickle.dump(likelihood_data, f)
                            
                        # likelihood_num += 1


                        llp_max = max(llp_max, temp)
                        llp_min = min(llp_min, temp)


                    llp = llp/len(data)
                    llp_model1 = llp_model1/len(data)
                    llp_model2 = llp_model2/len(data)
                    

                    pred = (pred1 + pred2)/2

                    

                loss = crit(pred, label)
                print(system, "\t\t {:.6f}".format(loss.item()), "\t\t {:.6f}".format(llp))

                
                loss_model1 = crit(pred1, label)
                print("gaussian_with_erf_loss_model1", "\t\t {:.6f}".format(loss_model1.item()), "\t\t {:.6f}".format(llp_model1))

                loss_model2 = crit(pred2, label)
                print("gaussian_with_erf_loss_model2", "\t\t {:.6f}".format(loss_model2.item()), "\t\t {:.6f}".format(llp_model2))