import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


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

def recommendationSystem(var_title):
    #csv파일 불러오기
    ratings = pd.read_csv('../../dataset/ratings.csv')
    movies = pd.read_csv('../../dataset/movies.csv')
    
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
    
    return predictions
    
######값넣을 부분##########
predictions = recommendationSystem(['Toy Story (1995)','Jumanji (1995)','Grumpier Old Men (1995)'])
print(predictions.loc[:,'title'].tolist())