import numpy as np
import scipy.linalg
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
import time

np.random.seed(0)

class Net(nn.Module):
	def __init__(self, k, input_shape):
		super(Net, self).__init__()
		self.fcn1 = nn.Linear(input_shape, k)
		self.fcn2 = nn.Linear(k, input_shape)

	def forward(self, i):
		V = self.fcn1(i)
		W = self.fcn2(F.sigmoid(V))
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

def train_net(net, train_data, regularizer = 0.1):
	net.train()
	if (torch.cuda.is_available()):
		net.cuda()
	optimizer = torch.optim.SGD(net.parameters(), lr = 1e-2, momentum = 0.5)
	loss = 0
	# import pdb;pdb.set_trace()
	for training_sample in train_data:
		optimizer.zero_grad()
		mask = map(lambda x: (x != 0) * 1., training_sample)
		i = np_variable(training_sample)
		if (torch.cuda.is_available()):
			i = i.cuda()
		p_o = net(i)
		mask = np_variable(mask)
		if (torch.cuda.is_available()):
			p_o = p_o.cuda()
			mask = mask.cuda()
		p_o.data.mul_(mask.data)
		error = net.se_loss(i, p_o)
		if (torch.cuda.is_available()):
			error.data.cpu().numpy()[0] += regularizer / 2. * (np.linalg.norm(net.fcn1.weight.data.cpu().numpy(), ord = 'fro') +\
				np.linalg.norm(net.fcn2.weight.data.cpu().numpy(), ord = 'fro'))
		else:
			error.data.numpy()[0] += regularizer / 2. * (np.linalg.norm(net.fcn1.weight.data.numpy(), ord = 'fro') +\
				np.linalg.norm(net.fcn2.weight.data.numpy(), ord = 'fro'))
		# print error
		error.backward()
		optimizer.step()
		loss += error.data[0]
	print 'average loss:', loss / len(train_data)

def test_net(net, train_data, test_data, user_AE):
	diff = 0.0
	for test_sample in test_data:
		if (user_AE):
			diff += np.abs(test_sample.rating - net(np_variable(train_data[test_sample.user_id])).data.numpy()[test_sample.item_id])
		else:
			diff += np.abs(test_sample.rating - net(np_variable(train_data[test_sample.item_id])).data.numpy()[test_sample.user_id])
	return diff / (4 * len(test_data))