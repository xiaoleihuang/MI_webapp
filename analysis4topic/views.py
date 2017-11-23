from django.shortcuts import render
from django.http import HttpResponse
from .infer_doc import LDA_Infer
import json

lda_infer = LDA_Infer('./resources/lda_model/ldamodel1.model','./resources/lda_model/lda_dict1.dict')

def index(request):
    return render(request, 'topics.html')
    # return HttpResponse("Hello, world. You're at the polls index.")

def topic_infer_calc(request):
    input_mode = None
    contents = []
    if request.method=='POST':
        input_mode = request.POST.get('input_mode')
        contents.append(request.POST.get('input_content'))
        if input_mode == 'multiple':
            count = 1
            while request.POST.get('input_content'+str(count)):
                contents.append(request.POST.get('input_content'+str(count)))
                count+=1
    else:
        return render(request, 'topics.html', {'error': 'The data is not valid and must be requested as POST.'})

    topic_dists = []
    for one_doc in contents:
        topic_dists.append(lda_infer.infer(one_doc))

    return render(request, 'topics.html', {
        'len': len(contents),
        'infer_topics': json.dumps(topic_dists)
    })