
import os
import pandas as pd
import numpy as np

from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
from django.conf import settings

from scipy.sparse.linalg import svds

def word2vecFn(m_list):
    # ========== 입력값 전처리 ========== #
    #print("print : ", os.getcwd())

    data = pd.read_csv(settings.MOVIELIST_USER_FILE_PATH)
    df_items = pd.read_csv(settings.MOVIES_FILE_PATH)

    # 사용자한테 영화명을3개 받을것
    # 영화명이 들어오면 영화아이디로
    #m_list = ['Toy Story (1995)','Jumanji (1995)','Grumpier Old Men (1995)']

    # m_list = [ ]

    # 숫자로 변환된 영화 아이디가 저장되는 리스트
    m_id_list = []
    # 숫자로 변환해 주기 위해 for문을 돌린다
    for m in m_list:
        m_id = df_items.loc[df_items['title'] == m]['movieId'].values
        m_id_list.append(m_id[0])


    # ========== 모델불러오기 ========== #
    # colab에서 movieList_users.csv를 통해 학습한 모델파일을 불러온다
    model = Word2Vec.load(settings.W2D_MODEL_FILE_PATH)


    # ========== word2vec 계산 ========== #

    word_vectors = model.wv
    vocabs = word_vectors.vocab.keys()
    word_vectors_list = [word_vectors[v] for v in vocabs]

    # 세번을 반복한다
    # 저장된 결과를 받는곳
    result_list = []


    # 아이디를 하나씩 받아 실행한다
    def most_similar(item):
        try:
            print("Similar of " + df_items[df_items['movieId'] == int(item)].iloc[0]['title'])
        except:
            print("Similar of " + item )
        
        return [[x, df_items[df_items['movieId'] == int(x[0])].iloc[0]['title']] for x in model.wv.most_similar(item)]


    for i in range(len(m_id_list)):
        predictions = most_similar(str(m_id_list[i]))
        result_list.append(predictions)



    # ========== 출력값 후처리 ========== #
    # 선택해서 리스트에 담아줌(정확도 높은순으로 유니크한값 추출)
    # 각각의 값들을 리스트로 만들어 합친다
    movieId_list = []
    similar_list = []
    title_list = []

    for i in range(9):
        r_id = result_list[0][i][0][0]
        r_s = result_list[0][i][0][1]
        r_title = result_list[0][i][1]
        
        movieId_list.append(r_id)
        similar_list.append(r_s)
        title_list.append(r_title)

    # 새로운 데이터 프레임에 값을 넣어 중복제거, 소팅한다
    predic_data = pd.DataFrame(columns=['movieId', 'similar', 'title'])
    predic_data['movieId'] = movieId_list
    predic_data['similar'] = similar_list
    predic_data['title'] = title_list
    predic_data = predic_data.sort_values(by='similar', na_position='last', ascending=False)

    # 결과를 CSV에 담아준다
    #predic_data.to_csv('predic_data.csv', index=0)

    #     print(result_list[0][1])    
    result_title = predic_data[:5]['title']
    result_similar = predic_data[:5]['similar']
    result_similar = round((result_similar), 3)

    print(result_title, result_similar)
    
    res = list(np.array(result_title.tolist()))

    return res

def recommend_movies(df_svd_preds, user_id, ori_movies_df, ori_ratings_df, num_recommendations=5):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
    user_row_number = user_id - 1 
    
    # 최종적으로 만든 pred_df에서 사용자 index에 따라 영화 데이터 정렬 -> 영화 평점이 높은 순으로 정렬 됌
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    
    # 원본 평점 데이터에서 user id에 해당하는 데이터를 뽑아낸다. 
    user_data = ori_ratings_df[ori_ratings_df.userId == user_id]
    
    # 위에서 뽑은 user_data와 원본 영화 데이터를 합친다. 
    user_history = user_data.merge(ori_movies_df, on = 'movieId').sort_values(['rating'], ascending=False)
    
    # 원본 영화 데이터에서 사용자가 본 영화 데이터를 제외한 데이터를 추출
    recommendations = ori_movies_df[~ori_movies_df['movieId'].isin(user_history['movieId'])]
    # 사용자의 영화 평점이 높은 순으로 정렬된 데이터와 위 recommendations을 합친다. 
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      

    return user_history, recommendations
    
def MFSVDFn(var_title):
    #csv파일 불러오기
    ratings = pd.read_csv(settings.RATINGS_FILE_PATH)
    movies = pd.read_csv(settings.MOVIES_FILE_PATH)
    
    #내 정보 추가
    var_userId=ratings['userId'].max()+1
    for temp in var_title:
        var_movieId=movies[movies['title']==temp]['movieId'].values[0]
        new_data_ratings = {'userId': var_userId,'movieId':var_movieId,'rating':5.0,'timestamp':0}
        ratings.append(new_data_ratings,ignore_index=True)
        
    #피벗테이블 만들기
    ratings_movies = pd.merge(ratings,movies, on='movieId')
    ratings_movies = ratings_movies.pivot_table(values='rating',index='userId',columns='movieId').fillna(0)
    
    #배열화
    matrix=np.array(ratings_movies)
    
    #열기준, 한 영화에 대한 모든 사용자들의 평균 평점
    user_ratings_mean = np.mean(matrix,axis=1) 
    #사용자-영화에 대한 사용자 평균 평점을 뺀 것,  그냥 2차원으로 만들어줌 그냥 더하면 안되나?
    matrix_user_mean = matrix-user_ratings_mean.reshape(-1,1)
    #SVD 학습
    U, sigma, Vt = svds(matrix_user_mean,k = 12)
    sigma = np.diag(sigma)
    
    #원본 행렬 복구
    svd_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt) + user_ratings_mean.reshape(-1,1)
    df_svd_preds = pd.DataFrame(svd_user_predicted_ratings,columns=ratings_movies.columns)
    
    already_rated, predictions = recommend_movies(df_svd_preds,2, movies, ratings,5)
    
    return predictions.loc[:,'title'].tolist()

