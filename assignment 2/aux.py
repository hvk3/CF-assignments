import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import time

np.random.seed(0)

class Net(nn.Module):
	def __init__(self, k, input_shape):
		super(Net, self).__init__()
		self.fcn1 = nn.Linear(input_shape, k)
		self.fcn2 = nn.Linear(k, input_shape)

	def forward(self, i):
		V = F.sigmoid(self.fcn1(i))
		W = self.fcn2(V)
		return W

	def se_loss(self, i, o):
		return torch.sum((i - o) * (i - o))

class info:
	def __init__(self, num_objects, object):
		self.num_objects = num_objects
		self.object = object

class item:
	def __init__(self, movie_id, movie_title, release_date, video_release_date, IMDb_URL, categories):
		self.movie_id = int(movie_id)
		self.movie_title = movie_title
		self.release_date = release_date
		self.video_release_date = video_release_date
		self.IMDb_URL = IMDb_URL
		self.categories = categories

class rating:
	def __init__(self, user_id, item_id, rating, timestamp):
		self.user_id = int(user_id)
		self.item_id = int(item_id)
		self.rating = int(rating)
		self.timestamp = int(timestamp)

class user:
	def __init__(self, user_id, age, gender, occupation, zip_code):
		self.user_id = int(user_id)
		self.age = int(age)
		self.gender = gender
		self.occupation = occupation
		self.zip_code = zip_code

def tensor_from_rating(rating):
	return torch.Tensor(np.array([int(rating.user_id - 1), int(rating.item_id - 1), int(rating.rating)]))

def read_file(filename, delimiter, class_type):
	inputs = []
	with open(filename, 'r') as f:
		for row in f.readlines():
			row = row.strip().split(delimiter)
			try:
				compressed_row = row[:5] + [row[5:]]
				inputs.append(class_type(*compressed_row))
			except:
				inputs.append(class_type(*row))
	return inputs

def get_delimiter_class(filename, dict):
	return dict[filename.strip().split('.')[-1]]

def generate_user_item_matrix(ratings, num_users, num_items):
	user_item_matrix = np.zeros((num_users, num_items))
	for rating in ratings:
		user_item_matrix[rating.user_id - 1][rating.item_id - 1] = rating.rating
	return user_item_matrix

def np_variable(var, requires_grad = False):
	return Variable(torch.from_numpy(np.array(var)), requires_grad = requires_grad).float()

def train_net(net, train_data, regularizer):
	optimizer = torch.optim.SGD(net.parameters(), lr = 1e-2)
	loss = 0
	if (torch.cuda.is_available()):
		net.cuda()
		kwargs = {'num_workers':4, 'pin_memory':True}
	else:
		kwargs = {}
	train_loader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True, **kwargs)
	for _, batch_data in enumerate(train_loader):
		masks = np_variable(np.array([map(lambda x: (x != 0) * 1., training_sample) for training_sample in batch_data]))
		if (torch.cuda.is_available()):
			batch_data = batch_data.cuda()
			masks = masks.cuda()
		batch_data = Variable(batch_data, requires_grad = True).float()
		V_sq_norm = np.linalg.norm(net.fcn1.weight.data.cpu().numpy(), ord = 'fro') ** 2
		W_sq_norm = np.linalg.norm(net.fcn2.weight.data.cpu().numpy(), ord = 'fro') ** 2
		optimizer.zero_grad()

		p_o = net(batch_data)
		p_o.data.mul_(masks.data)
		error = net.se_loss(batch_data, p_o)
		error.data += regularizer / 2. * (V_sq_norm + W_sq_norm)
		error.backward()
		optimizer.step()

def test_net(net, train_data, test_data, user_AE):
	diff = 0.0
	if (torch.cuda.is_available()):
		net.cuda()
		kwargs = {'num_workers':4, 'pin_memory':True}
	else:
		kwargs = {}
	test_data = map(lambda x: tensor_from_rating(x), test_data)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16, shuffle = True, **kwargs)
	for batch_data in test_loader:
		if (user_AE):
			corr_vars = np_variable(np.array(map(lambda x: train_data[int(x[0])], batch_data)))
		else:
			corr_vars = np_variable(np.array(map(lambda x: train_data[int(x[1])], batch_data)))
		masks = torch.zeros(corr_vars.size())
		for i in xrange(len(batch_data)):
			try:
				if (user_AE):
					masks[i][int(batch_data[i][1])] = 1.
				else:
					masks[i][int(batch_data[i][0])] = 1.
			except:
				import pdb;pdb.set_trace()
		gt_ratings = np_variable(np.array(map(lambda x: x[2], batch_data)))
		if (torch.cuda.is_available()):
			masks = Variable(masks).float().cuda()
			corr_vars = corr_vars.cuda()
			gt_ratings = gt_ratings.cuda()
		predictions = net(corr_vars)
		predictions.data.clamp_(1, 5)
		predictions.data.mul_(masks.data)
		final_predictions = torch.cumsum(predictions, dim = 1)[:, -1]
		errors = torch.abs(final_predictions.data - gt_ratings.data)
		diff += torch.sum(errors)
	return diff / len(test_data)
