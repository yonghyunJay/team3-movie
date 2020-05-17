from django.conf.urls import url 
from . import views
from django.urls import path

urlpatterns = [
    url(r'^$',views.index, name = 'index'),
    path('word2vec/', views.word2vec),
    path('word2vec/result/', views.word2vec_result),
]