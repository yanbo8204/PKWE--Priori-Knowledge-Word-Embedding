import os
import json
import scipy
import numpy as np
import networkx as nx
import pickle
import matplotlib
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

import dgl
import dgl.function as fn
from dgl.data.utils import load_graphs

from copy import deepcopy
import matplotlib.pyplot as plt
from pylab import show
from itertools import filterfalse
from collections import Counter
import sys

import utils
from utils import cosine_sim, l2norm, sequence_mask, lemmatize_all, make_embeddings, cosine_sim
from Tree_LSTM import treelstm, gru

from sklearn import preprocessing
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)

num_limit = 130000
link_limit = 200000

def comb_update(left, right, scope):
	prev_left_range = None
	prev_right_range = None
	new_left = torch.zeros(left.size(0), left.size(1) - 1)
	new_right = torch.zeros(right.size(0), right.size(1) - 1)
	if left.size(1) > 1:
		prev_left_range = right[left.eq(scope[:, 0].view(-1, 1))]
		prev_right_range = left[right.eq(scope[:, 1].view(-1, 1))]

		mask = left.eq(scope[:, 0].view(-1, 1))
		temp = list(torch.where(mask == True))
		mask[temp] = False
		temp[1] += 1
		mask[temp] = True
		new_left = left[~mask].view(left.size(0), -1)

		mask = right.eq(scope[:, 1].view(-1, 1))
		temp = list(torch.where(mask == True))
		mask[temp] = False
		temp[1] -= 1
		mask[temp] = True
		new_right = right[~mask].view(right.size(0), -1)
		#left.remove(left[left.index(int(scope[0])) + 1])
		#right.remove(right[right.index(int(scope[1])) - 1])

	return new_left, new_right, prev_left_range, prev_right_range

def graph_update(g, node=False, number = None, embedd=None, edge=None, bidirection=False):
	if node == True:
		g.add_nodes(number)

	if edge != None:
		if bidirection == True:
			g.add_edges(edge[0], edge[1])
			g.add_edges(edge[1], edge[0])
		else:
			g.add_edges(edge[0], edge[1])

	return g


class GMesReduce(nn.Module):
	def __init__(self):
		super(GMesReduce, self).__init__()
		self.list_leng = 0
		self.rep = []
		self.loss = 0
		self.count = 0
		self.lr = 1

	def forward(self, g, mode):
		print(mode)
		if mode == "mean":
			while True:
				a = g.ndata['repre'].pow(2).sum()
				for _ in range(1000):
					g.update_all(message_func=fn.copy_src(src='repre', out='msg'), 
						reduce_func=fn.mean(msg='msg',out='repre')) 
				print(abs(a - g.ndata['repre'].pow(2).sum()))
				if abs(a - g.ndata['repre'].pow(2).sum()) <= 0.1:
					break
			return g.ndata.pop('repre')

		self.rep = g.ndata['repre'].detach()
		self.list_leng = self.rep.size(0)

		for _ in range(20):
			self.loss = 0
			self.count = 0
			g.update_all(message_func=fn.copy_src(src='repre', out='msg'), 
				reduce_func=self.gcn_reduce) #fn.mean(msg='msg',out='repre')
			g.ndata['repre'] = g.ndata['repre'].detach()
			self.rep = g.ndata['repre'].detach()
			self.list_leng = self.rep.size(0)
			print("loss: ", self.loss/self.count)

		return g.ndata.pop('repre')

	def gcn_message(self, edges):
		# batch of edges
		return {'msg': edges.src['repre'] }

	def gcn_reduce(self, nodes):
		edges = nodes.mailbox['msg'].size(1)
		
		if edges == 1:
			return {'repre': nodes.data['repre']}
		elif edges > 5000:
			return {'repre': nodes.mailbox['msg'].mean(1).detach()}

		batch = nodes.mailbox['msg'].size(0)

		nodes.data['repre'] = torch.autograd.Variable(nodes.data['repre'].detach(), requires_grad=True)
		nodes.mailbox['msg'] = nodes.mailbox['msg'].detach()

		sim = nodes.mailbox['msg'].mean(1,keepdim=True).bmm(nodes.data['repre'].unsqueeze(-1)).squeeze(2)

		neg_samp = self.rep[torch.randint(2,self.list_leng,(batch,10))].detach()
		ratio = torch.cat([sim, neg_samp.bmm(nodes.data['repre'].unsqueeze(-1)).squeeze(2)], 1)
		#'''.min(1)[0].unsqueeze(-1)'''
		assert ratio.size(1) == 11 
		loss = (torch.log(torch.exp(ratio - ratio.max(1,keepdim=True)[0]).sum(1))
			).sum() - (ratio[:,0] - ratio.max(1)[0]).sum()
		
		self.loss += float(loss)
		self.count += batch
		#del rage
		loss.backward()

		assert torch.isnan(nodes.data['repre'].grad.data.sum()) != True

		clip_grad_norm_(nodes.data['repre'], 2.0)
		#print(nodes.data['repre'].grad.data)

		return {'repre': (nodes.data['repre'] - self.lr*nodes.data['repre'].grad.data).detach()}


class Graph(nn.Module):
	def __init__(self, opt):
		super(Graph, self).__init__()
		vocab = pickle.load(open(os.path.join(opt.data_path, 'vocab.pkl'), 'rb'))
		self.idx2node = {():[0,1], ('<unk>',):[1,1], ('<start>',):[2,0], 
						('<end>',):[3,0], ('.',):[4,0], (',',):[5,0]}

		self.idx2iamge = {}
		self.vocab = vocab
		self.node_length = 6

		self.g = dgl.DGLGraph()
		self.g = graph_update(self.g, node=True, number = 6)

		self.embedding = opt.word_dim

		self.MReduce = GMesReduce()
		self.representation = torch.randn(6, self.embedding).cuda()

		self.edges = {}#{(1,1):0}
		self.pre_img_list = []

		if torch.cuda.is_available():
			self.g = self.g.to('cuda:0')
			self.representation = self.representation.cuda()
			cudnn.benchmark = True


	def graph_update(self, img_embedd, targets, lengths, img_ids, captions=None):
		"""adding edge and node into graph"""
		count = 0	
		index_img_embedd = []
		left_image_node, right_image_node = [], []
		src_edge, det_edge = [], []

		minus = list(set(img_ids) ^ (set(img_ids) & set(list(self.idx2iamge.keys()))))
		count += len(minus)
		for i in minus:
			self.idx2iamge[i] = self.node_length
			self.pre_img_list.append(i)
			index_img_embedd.append(img_ids.index(i))

			self.node_length += 1
		
		self.representation = torch.cat((self.representation, img_embedd[index_img_embedd]))

		new_word = self.new_word_select(captions, lengths, stage='build_graph')
		num_new = Counter(new_word)[1]
		self.representation = torch.cat((self.representation, 
									torch.randn(num_new,self.embedding).cuda()))
		count += num_new

		for i in range(len(img_ids)):
			img_number = self.idx2iamge[img_ids[i]]
			for j in range(lengths[i]):
				cur_node = tuple([captions[i][j]])
				number = self.idx2node[cur_node]

				if (img_number, number[0]) not in self.edges:
					self.edges[(img_number, number[0])] = 0
					src_edge.append(img_number)
					det_edge.append(number[0])
					self.idx2node[cur_node][1] += 1

		self.g = graph_update(self.g, node=True, number = count)
		self.g = graph_update(self.g, edge=[src_edge, det_edge], bidirection = False)
		self.g = graph_update(self.g, edge = [left_image_node, right_image_node], bidirection = False)

		self.representation = self.representation.detach()

	
	def new_word_select(self, word_list, length, stage = None):
		idx_rep = []

		for j in range(len(word_list)):
			idx_rep_sig = []
			for i in range(length[j]):
				word = tuple(word_list[j][i:i+1])
				index_graph = self.idx2node.get(word, -1)
				if index_graph == -1:
					idx_rep_sig.append(1)
					if stage != None:
						self.idx2node[word] = [self.node_length,0]
						self.node_length += 1
				else:
					idx_rep_sig.append(index_graph[0])

			idx_rep_sig.extend([0]*(int(length[0]-length[j])))
			idx_rep.extend(idx_rep_sig)

		return idx_rep


	def mask_state(self, state, index):
		mask_left = sequence_mask(index[:,0], max_length=state.size(1))
		mask_right = sequence_mask(index[:,1] + 1, max_length=state.size(1))
		mask_toge = (~mask_left) * mask_right

		return mask_toge


	def forward(self, mode):
		self.g.ndata['repre'] = self.representation.clone().detach()
		self.representation = None
		self.representation = self.MReduce(self.g, mode).detach()
			

