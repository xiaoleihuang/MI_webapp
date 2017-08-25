from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, render_to_response
import sys
if sys.version_info[0] == 2:
	from unidecode import unidecode

#try:
#	import simplejson as json
#except:
import json
import theano
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

def index(request):
	return render(request, 'att_viz/index.html')

def test(request):
	# test json
	import simplejson as json
	spaceCircles = [30, 70, 110]
	test = [{'word':'alcohol', 'value':20}, {'word':'like', 'value':10},{'word':'hello', 'value':1}]
	test = json.dumps(test)
	print(test)
	return render(request, 'att_viz/test.html',{'spaceCircles':spaceCircles, 'test': test})

from .mylib import myconfig
from .forms import AttentionForm
from .mylib import model_helper, data_helper

# load ini properties
model_config = model_helper.load_model_config()

# load keras models and tokenizer
keras_model, keras_model_context, keras_tokenizer = model_helper.load_keras_resources()

if keras_model != None:
	rnn_layer_func_content = model_helper.get_rnn_func(keras_model)

if keras_model_context != None:
	rnn_layer_func_ctxt = model_helper.get_rnn_func(keras_model_context, input_mode=2)

# load configuration
config_dict = myconfig.BaseConfiguration()

def attention_form_calc(request):
	input_mode = None
	context = None
	content = None
	if request.method=='POST':
		#att_form = AttentionForm(request.POST)
		input_mode = request.POST.get('input_mode')
		context = request.POST.get('input_context')
		content = request.POST.get('input_content')
	else:
		return render(request, 'att_viz/index.html', {'error': 'The data is not valid and must be requested as POST.'})

	# check the data and models are validate or not
	error = model_helper.validate_inputs(
		input_mode, content, context, keras_model, keras_model_context)
	if error:
		return render(request, 'att_viz/index.html', {'error': error})

	# preprocessing the dataset
	input_mode = int(input_mode)
	content = content.strip()
	rnn_layer_funcs = rnn_layer_func_content # for content mode
	keras_model_input = keras_model

	if sys.version_info[0] == 2:
		content = unidecode(content)
	content_proc = data_helper.preproc_data([content])
	content_proc_idx = data_helper.proc_pipeline(
		content_proc, keras_tokenizer, config_dict.seq_max_len)

	context_proc_idx=None
	if input_mode == 2: # for the context
		if sys.version_info[0] == 2:
			context = unidecode(context)
		context = context.strip()
		context_proc = data_helper.preproc_data([context])
		context_proc_idx = data_helper.proc_pipeline(context_proc, keras_tokenizer, config_dict.seq_max_len)

		rnn_layer_funcs = rnn_layer_func_ctxt # contextual mode functions
		keras_model_input = keras_model_context

	pred_probs,pred_cls,att_weights,att_weights_ctxt=model_helper.cal_pipeline(
            content_proc_idx, keras_model_input, input_context=context_proc_idx,
            input_mode=input_mode, rnn_layer_func_list=rnn_layer_funcs)

	# process probability for each catgory
	sum_probs = sum(pred_probs)# normalize for sum to 1
	pred_probs = [round(itm/sum_probs*100,2) for itm in pred_probs]

	# start to mapping words and its attention weights
	content_words = content_proc[0].split()
	att_content_list = []
	for word in content_words:
		if word in keras_tokenizer.word_index:
			att_content_list.append({'word': word,'value':str(att_weights.get(keras_tokenizer.word_index[word],0))})
		else:
			att_content_list.append({'word': word,'value': '0'})

	# contextual mode for mapping words and its attention weights
	if input_mode == 2:
		context_words = context_proc[0].split()
		att_ctxt_list = []
		for word in context_words:
			if word in keras_tokenizer.word_index:
				att_ctxt_list.append({'word': word,'value':str(att_weights_ctxt.get(keras_tokenizer.word_index[word],0))})
			else:
				att_ctxt_list.append({'word': word,'value': '0'})

		return render(request, 'att_viz/index.html', {
			'form': input_mode,
			'pred_probs':pred_probs, 'pred_cls':pred_cls,
			'content_tuple':json.dumps(att_content_list),
			'context_tuple':json.dumps(att_ctxt_list),
		})

	# content-only mode
	return render(request, 'att_viz/index.html', {
		'form': input_mode,
		'pred_probs':pred_probs, 'pred_cls':pred_cls,
		'content_tuple':json.dumps(att_content_list),
	})
