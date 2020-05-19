from django.conf.urls import url 
from . import views
from django.urls import path

urlpatterns = [
    url(r'^$',views.index,name = 'index'),
    path('recommendMovies1/', views.get_recommend_movies_1, name='get_recommend_movies_1'),
    path('recommendMovies2/', views.get_recommend_movies_2, name='get_recommend_movies_2'),
    path('recommendMovies3/', views.get_recommend_movies_3, name='get_recommend_movies_3'),
]