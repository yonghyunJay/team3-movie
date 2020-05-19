from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from .util import random_poster_urls, get_poster_urls_to_list, get_naver_id_to_list
from .movie_model import word2vecFn, MFSVDFn
from .knn_model import KNNFn
from django.views.decorators.csrf import csrf_exempt
import json

def index(request):
    template = loader.get_template('movieApp/index.html')
    context = {
        'latest_question_list': "test",
    }
    #return HttpResponse(template.render(context, request))
    res_dict = random_poster_urls(10)
    return render(request, 'movieApp/index.html', {'res_dict' : res_dict})

def get_recommend_movies_1(request):
    if request.method == 'POST':
        recv_movies_list = request.POST.getlist('nameList[]')
        #print(recv_movies_list)

        # ToDo. 추천 모델 호출
        # recv_movies_list -> (model input)
        # (model output) -> outputList
    
        response_data = {}
        outputList = word2vecFn(recv_movies_list)
        
        response_data['name'] = outputList
        response_data['naver_url'] = get_naver_id_to_list(outputList)
        response_data['poster_url'] = get_poster_urls_to_list(outputList)

        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )

def get_recommend_movies_2(request):
    if request.method == 'POST':
        recv_movies_list = request.POST.getlist('nameList[]')
        #print(recv_movies_list)

        # ToDo. 추천 모델 호출
        # recv_movies_list -> (model input)
        # (model output) -> outputList
    
        response_data = {}
        outputList = MFSVDFn(recv_movies_list)
        
        response_data['name'] = outputList
        response_data['naver_url'] = get_naver_id_to_list(outputList)
        response_data['poster_url'] = get_poster_urls_to_list(outputList)

        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )

def get_recommend_movies_3(request):
    if request.method == 'POST':
        recv_movies_list = request.POST.getlist('nameList[]')
        #print(recv_movies_list)

        # ToDo. 추천 모델 호출
        # recv_movies_list -> (model input)
        # (model output) -> outputList
    
        response_data = {}
        outputList = KNNFn(recv_movies_list)
        
        response_data['name'] = outputList
        response_data['naver_url'] = get_naver_id_to_list(outputList)
        response_data['poster_url'] = get_poster_urls_to_list(outputList)

        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_type="application/json"
        )
