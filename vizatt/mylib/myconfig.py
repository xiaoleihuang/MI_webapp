import os
class BaseConfiguration:
	def __init__(self):
		self.learning_rate = 0.01
		self.training_iters = 40000
		self.vocabulary_size = 10000
		self.batch_size = 50
		self.display_step = 1
		self.val_step = 50
		self.utterances = 10
		# Network Parameters
		self.seq_max_len = 100 # Sequence max length
		self.seq_min_len = 5
		self.embedding_size = 200
		self.n_hidden = 100 # hidden layer num of features
		self.n_classes = 2 # linear sequence or not
		#self.num_layers = 1
		#self.keep_prob = 1
		#self.debug_sentences = True
		self.path = os.getcwd()
		self.filenames = filter(lambda x: x[-1]=='5' ,os.listdir('./'))

	def printConfiguration(self):
		# print configuration
		print('---------- Configuration: ----------')
		print('learning_rate', self.learning_rate)
		print('training_iters', self.training_iters)
		print('batch_size', self.batch_size)
		print('vocab_size', self.vocabulary_size)
		# Network Parameters
		print('seq_max_len', self.seq_max_len) # Sequence max length
		print('seq_min_len', self.seq_min_len)
		print('embedding_size', self.embedding_size)
		print('n_hidden', self.n_hidden) # hidden layer num of features
		print('n_classes', self.n_classes) # linear sequence or not
		print('utterances in a sequence', self.utterances)
		#print('dataset size', self.file_len)
		# print('num_layers', self.num_layers)
		# print('keep_prob (dropout = 1-keep_prob)', self.keep_prob)
		print('------------------------------------')
