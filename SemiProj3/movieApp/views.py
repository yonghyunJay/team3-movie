from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from movieApp.templates.movieApp.juhyun.word2vec import word2vecFn
 
def index(request):
    template = loader.get_template('movieApp/index.html')
    context = {
        'latest_question_list': "test",
    }
    return HttpResponse(template.render(context, request))


# 20년 05월 17일 수정 (주현)
# 8000/word2vec 을 입력하면 word2vec.py 파일이 실행되고 터미널에 결과값이 보여진다
# word2vec.py 에 임의의 3개 영화명 넣어놓음(장고에서 입력받아 내보내는 형태로 수정해야함)
# word2vec 입력받는곳
def word2vec(request):
    predic_data = word2vecFn()
    return render(
        request, 
        'movieApp/juhyun/test.html', 
        {
        'predic_data': predic_data
        })

# word2vec 결과 받는곳
# 8000/word2vec/result 을 입력하면 함수실행되고 결과가 보임
def word2vec_result(request):
    predic_data = word2vecFn()
    return render(
        request, 
        'movieApp/juhyun/result.html', 
        {
        'predic_data': predic_data
        })