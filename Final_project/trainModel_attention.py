from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Merge
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from numpy import array
from attention_decoder import AttentionDecoder

SINGLE_ATTENTION_VECTOR = False
#import pydot 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
 
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	batch_size = 1
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):

			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[:i+1]
			# in_seq, out_seq = seq[:i], seq[i]
			#print(len(in_seq),len(out_seq))
			# pad input sequence
			in_seq = in_seq[::-1]
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			in_seq = in_seq[::-1]
			#in_seq =np.reshape(in_seq,(batch_size,max_length))
			#print (in_seq)
			# encode output sequence
			out_seq = out_seq[::-1]
			out_seq = pad_sequences([out_seq], maxlen=max_length)[0]
			out_seq = out_seq[::-1]
			#print(out_seq)
			out_seq = to_categorical([out_seq],num_classes=vocab_size)#[0]
			out_seq =np.reshape(out_seq,(max_length,vocab_size))
			#print(out_seq.shape)
			#print(out_seq[0][0])
			#out_seq = 
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq) ### TODO: FIGURE OUT HOW TO MASK THE ZEROS.
	#print(len(X2[0]),len(y))
	return array(X1), array(X2), array(y)
 

# def attention_3d_block(inputs,max_length):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[1]) ### DIMENSIONS NEED TO MATCH
#     print(input_dim)
#     max_length = 256
#     #a = Permute((2, 1))(inputs)
#     #a = Reshape((input_dim, max_length))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(max_length, activation='softmax')(inputs)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     #a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = merge([inputs, a], name='attention_mul', mode='mul')
#     return output_attention_mul
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[1])
    # 1 D convolution layer along z dim
    # 1D convolution along embed dimension
    # Repeated h vector as many local feats in v
    #
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	# Inputs1 is image features
	EMBDIM = 256
	ZDIM = 512
	LSTMDIM = 256

	inputs1 = Input(shape=(7,7,512,))
	inputs1_reshape = Reshape((49,512,),name='inputReshape')(inputs1)
	fe1 = GlobalAveragePooling1D(name='avgIn')(inputs1_reshape)
	fe1 = Dropout(0.5)(fe1)
	fe2 = Dense(EMBDIM, activation='relu')(fe1) # Similar to Vg

	## Project input to to zspace (512 I think)
	spatial_feats = Conv1D(ZDIM,1,padding='same',activation='relu',name='reshape_emb')(inputs1_reshape) 
	#similar to Vi
	spatial_feats_drop = Dropout(0.5)(spatial_feats)
	s_feats_emb = Conv1D(EMBDIM,1,padding='same',activation='relu',name='sf_emb')(spatial_feats_drop)
	fe2_rep = RepeatVector(max_length)(fe2)
	#print('fe2 rep shape: ',fe2_rep.shape)
	
	## Inputs2 is past words generated.
	inputs2 = Input(shape=(max_length,))
	
	se1 = Embedding(vocab_size, EMBDIM,input_length=max_length)(inputs2) # similar to wemb
	se1_act = Activation('relu')(se1)
	#print('se1 shape: ',se1.shape)
	se2 = Dropout(0.5)(se1_act) 
	#print('se2 shape: ',se2.shape)

	#Merge fe2_rep and then call LSTM.
	x = Concatenate(name='lstm_input')([fe2_rep,se2])
	#print('fe2 shape: ',fe2_rep.shape)
	print('x shape: ', x.shape)
	se3 = LSTM(LSTMDIM,return_sequences=True)(x) ### Does masking zero come back into this
	#print('se3 shape', se3.shape)

	#h_out_lin
	se4_lin = Conv1D(ZDIM,1,activation='tanh',name='zh_linear',padding='same')(se3)
	print('se4 shape: ',se4_lin.shape)
	se4_lin_dr = Dropout(0.5)(se4_lin)
	se4_emb = Conv1D(EMBDIM,1,name='se4_emb',padding='same')(se4_lin_dr)
	se4_emb_rep = TimeDistributed(RepeatVector(49),name='rep_se4_emb')(se4_emb)
	print('se4_emb_rep shape: ',se4_emb_rep.shape)

	se5_lin = TimeDistributed(RepeatVector(max_length),name='z_lin')(spatial_feats_drop)
	se5_emb = TimeDistributed(RepeatVector(max_length),name='se5_emb')(s_feats_emb)
	se5_lin = Permute((2,1,3))(se5_lin)
	se5_emb = Permute((2,1,3))(se5_emb)
	print(se5_emb.shape)

	lstm_attn_merge = Add(name='lstm_attn_merge')([se4_emb_rep,se5_emb])
	lstm_attn_merge = Dropout(0.5)(lstm_attn_merge)
	lstm_attn_merge = TimeDistributed(Activation('tanh'),name='attnInActivate')(lstm_attn_merge)

	#print('lstm_attn:',lstm_attn_merge.shape)
	attn = TimeDistributed(Conv1D(1,1,padding='same'),name='attn')(lstm_attn_merge)
	attn = Reshape((max_length,49,))(attn) ### NOT ENTIRELY SURE THIS IS RIGHT.
	attn = TimeDistributed(Activation('softmax'),name='softmaxAttn')(attn)
	print('attn shape: ',attn.shape)
	attn = TimeDistributed(RepeatVector(ZDIM),name='repAttn')(attn)
	attn = Permute((1,3,2))(attn)

	w_context = Multiply()([attn,se5_lin])
	sumpool = Lambda(lambda x: K.sum(x,axis=-2),output_shape=(ZDIM,))
	se6 = TimeDistributed(sumpool,name='sumpooled')(w_context)
	attn_out = Add()([se4_lin_dr,se6])
	decoder = TimeDistributed(Dense(EMBDIM,activation='tanh'),name='decoder')(attn_out)
	decoder = Dropout(0.5)(decoder)

	## Combination of models.
	print('decoder Shape: ', decoder.shape)
	#poolDecoder = GlobalAveragePooling1D()(decoder) ### THIS IS BY NO MEANS CORRECT HERE
	#print(poolDecoder.shape)
	#outputs = TimeDistributed(Dense(vocab_size, activation='softmax',name='output_d'),name='output_t')(decoder)
	outputs = Dense(vocab_size, activation='softmax',name='output_d')(decoder)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model
 
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo) ### TODO: MODIFY TO BE SEQUENCE-TO-SEQUENCE
			#print([[in_img, in_seq], out_word])
			yield [[in_img, in_seq], out_word] ### THIS RETURN IS PROBABLY A POINT OF FIXING TOO
 

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features_diff.pkl', train)
# print('length photos: ',train_features)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)#
print('Description Length: %d' % max_length)

### PROBABLY WRONG
#max_length = 1
 
# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	inputs, outputs = next(generator)
	print(inputs[0].shape)
	print(inputs[1].shape)
	print(outputs.shape)


	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')