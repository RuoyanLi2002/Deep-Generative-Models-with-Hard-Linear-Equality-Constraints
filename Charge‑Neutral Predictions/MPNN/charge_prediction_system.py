# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 22, 2019
# // PURPOSE     : main function.
# // SPECIAL NOTES: Uses data_handling.py and model.py
# // ===============================
# // Change History: 1.0: initial code: wrote and tested.
# // Change History: 1.5: Added: minibatches, training/testing/validation split
# // Change History: 1.6: Troubleshooting: fixed bugs related to incorrect handling of minibatches
# // Change History: 2.0: Chagnes: Interface for wrapper for testing and evaluation
# //
# //==================================
__author__ = "Ali Raza"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = ""
__version__ = "1.0"
__maintainer__ = "ali raza"
__email__ = "razaa@oregonstate.edu"
__status__ = "done"

from model import *
from matplotlib import pyplot as plt
import copy
import torch
from tqdm import tqdm, trange


def erf_loss(mu, sigma, b, data, device):
	# total_loss = torch.tensor([0.0]).to(device)
	# for i in range(0, data.num_graphs):
	# 	temp_mu = mu[data.batch == i] + sigma[data.batch == i]*(k - torch.sum(mu[data.batch == i]))/torch.sum(sigma[data.batch == i])
	# 	temp_sigma = sigma[data.batch == i] - torch.square(sigma[data.batch == i])/torch.sum(sigma[data.batch == i])
	# 	temp_b = b[data.batch == i]
	# 	loss = (2*temp_sigma/torch.pi)**(1/2) * torch.exp(-((temp_mu-temp_b)**2) / (2*temp_sigma)) + (temp_mu - temp_b) * torch.erf((temp_mu-temp_b)*((2*temp_sigma)**(-1/2)))
	# 	each_batch_loss = torch.sum(loss)
	# 	total_loss += each_batch_loss
	# return total_loss
	
	loss = (2*sigma/torch.pi)**(1/2) * torch.exp(-((mu-b)**2) / (2*sigma)) + (mu - b) * torch.erf((mu-b)*((2*sigma)**(-1/2)))
	return torch.mean(loss)

def multivariate_mean_variance(means, sigmas, device):
    n = len(sigmas)

    A = torch.diag(sigmas[:-1])
    B = torch.ones(n-1, n-1).double().to(device) * torch.pow(sigmas[-1], -1)

    covariance_matrix = A - 1/(1 + torch.trace(torch.matmul(B, A))) * torch.matmul(A ,torch.matmul(B, A))

    c = (k - means[-1])/sigmas[-1]
    reduced_mean = torch.matmul(covariance_matrix, torch.ones(n-1).to(device)*c + torch.div(means[:-1], sigmas[:-1]))

    return reduced_mean, covariance_matrix

def likelihood_loss(mu, sigma, b, data, device):
	total_likelihood = torch.tensor([0.0]).to(device)

	for i in range(0, data.num_graphs):
		Mean, Sigma = multivariate_mean_variance(mu[data.batch == i], sigma[data.batch == i], device)
		distrib = MultivariateNormal(loc=Mean, covariance_matrix=Sigma)
		ground_truth = b[data.batch == i]
		total_likelihood += -distrib.log_prob(ground_truth[:-1])
	
	return total_likelihood
	 

def charge_prediction_system(train_loader,valid_loader,test_loader,NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE,train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, patience, learning_rate = 0.005, crit = torch.nn.L1Loss(), evaluation_mode = False, evaluation_num = 0, device = torch.device('cuda:0')):
	# initializing the model	
	if system == 'gaussian_cor':
		# print(">>> gaussian_correction_model")
		model = Net_gaussian_correction(NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE).to(device)
	elif system == 'gaussian_cor_with_sampling':
		# print(">>> gaussian_correction_model_with_sampling")
		model = Net_gaussian_correction_with_sampling(NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE).to(device)
	else:
		# print(">>> gaussian_correction_model_with_erf_loss")
		model = Net_gaussian_correction_with_erf_loss(NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	model = model.double()

	train_total_loss = []
	valid_total_loss = []
	test_total_loss = []
	min_valid_loss = float("inf")
	rebound = 0 # number of epochs that validation loss is increasing
	for epoch in trange(MAX_EPOCHS):
		model.train()
		loss_all = 0
		for data in train_loader:
			data = data.to(device)
			label = data.y.to(device)
			optimizer.zero_grad()

			if system == 'gaussian_cor':
				pred, _, _, _ = model(data)
				loss = crit(pred, label)
			elif system == 'gaussian_cor_with_sampling':
				pred = model(data, True)
				loss = crit(pred, label)
			elif system == "gaussian_with_erf_loss":
				_, _, mean_bar, sigma_bar = model(data)
				loss = erf_loss(mean_bar, sigma_bar, label, data, device)
			else:
				mu, sigma, _, _ = model(data)
				loss = likelihood_loss(mu, sigma, label, data, device)
			
			loss.backward()
			loss_all += loss.item()
			optimizer.step()

		loss_epoch = loss_all / train_data_size

		# # evaluating model
		# model.eval()
		# loss_all = 0
		# with torch.no_grad():
		# 	for data in train_loader:
		# 		data = data.to(device)
		# 		label = data.y.to(device)

		# 		if system == 'gaussian_cor':
		# 			pred, _, _, _ = model(data)
		# 			loss = crit(pred, label)
		# 		elif system == 'gaussian_cor_with_sampling':
		# 			pred = model(data, False)
		# 			loss = crit(pred, label)
		# 		elif system == "gaussian_with_erf_loss":
		# 			_, _, mean_bar, sigma_bar = model(data)
		# 			loss = erf_loss(mean_bar, sigma_bar, label, data, device)
		# 		else:
		# 			mu, sigma, _, _ = model(data)
		# 			loss = likelihood_loss(mu, sigma, label, data, device)

		# 		loss_all += data.num_graphs * loss.item()

		# train_acc = loss_all / train_data_size
		# train_total_loss.append(train_acc)

		# evaluating valid dataset
		model.eval()
		loss_all = 0
		with torch.no_grad():
			for data in valid_loader:
				data = data.to(device)
				label = data.y.to(device)

				if system == 'gaussian_cor':
					pred, _, _, _ = model(data)
					loss = crit(pred, label)
				elif system == 'gaussian_cor_with_sampling':
					pred = model(data, False)
					loss = crit(pred, label)
				elif system == "gaussian_with_erf_loss":
					_, _, mean_bar, sigma_bar = model(data)
					loss = erf_loss(mean_bar, sigma_bar, label, data, device)
				else:
					# mu, sigma, _, _ = model(data)
					# loss = likelihood_loss(mu, sigma, label, data, device)
					_, _, mean_bar, sigma_bar = model(data)
					loss = erf_loss(mean_bar, sigma_bar, label, data, device)
				
				# loss_all += data.num_graphs * loss.item()
				loss_all += loss.item()
		valid_acc = loss_all / valid_data_size
		valid_total_loss.append(valid_acc)

		# evaluating test dataset
		# loss_all = 0
		# with torch.no_grad():
		# 	for data in test_loader:
		# 		data = data.to(device)
		# 		label = data.y.to(device)

		# 		if system == 'gaussian_cor':
		# 			pred, _, _, _ = model(data)
		# 			loss = crit(pred, label)
		# 		elif system == 'gaussian_cor_with_sampling':
		# 			pred = model(data, False)
		# 			loss = crit(pred, label)
		# 		elif system == "gaussian_with_erf_loss":
		# 			_, _, mean_bar, sigma_bar = model(data)
		# 			loss = erf_loss(mean_bar, sigma_bar, label, data, device)
		# 		else:
		# 			mu, sigma, _, _ = model(data)
		# 			loss = likelihood_loss(mu, sigma, label, data, device)
				
		# 		loss_all += data.num_graphs * loss.item()
		# test_acc = loss_all / test_data_size
		# test_total_loss.append(test_acc)
			
		if valid_acc <= min_valid_loss: # keep tracking of model with lowest validation loss
			torch.save(model.state_dict(), './results/loss_iteration_' + str(iteration)+'_system_' + system + '.pth')
			min_valid_loss = valid_acc
			model_min = copy.deepcopy(model)
			rebound = 0
		else:
			rebound += 1

		if rebound > patience: # early stopping criterion
			break

	return model_min, train_total_loss, valid_total_loss, test_total_loss
