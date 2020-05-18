
# usage :: 'python model/mirae/knn_recommender.py --movie_name "Iron Man" --top_n 10' # 팀 폴더 
# usage :: 'python knn_recommender.py --movie_name "Iron Man" --top_n 10' #개인 폴더
import os
import time
import gc
import argparse
# import total
import numpy as np
# data science imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# utils import
from fuzzywuzzy import fuzz



class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with KNN implmented by sklearn
    """
    def __init__(self, path_movies, path_ratings):
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = 0
        self.user_rating_thres = 0
        self.model = NearestNeighbors()

    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        """
        set rating frequency threshold to filter less-known movies and
        less active users
        Parameters
        ----------
        movie_rating_thres: int, minimum number of ratings received by users
        user_rating_thres: int, minimum number of ratings a user gives
        """
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        """
        set model params for sklearn.neighbors.NearestNeighbors
        Parameters
        ----------
        n_neighbors: int, optional (default = 5)
        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        metric: string or callable, default 'minkowski', or one of
            ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        n_jobs: int or None, optional (default=None)
        """
        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors, #가장 가까운 이웃 몇개 고를껀지 
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _prep_data(self):
        """
        prepare data for recommender
        1. movie-user scipy sparse matrix
        2. hashmap of movie to row index in movie-user scipy sparse matrix #영화와 인덱스간의 해쉬맵
        """
        # read data
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        # filter data
        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))  # 인기영화위주로 
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # pivot and create movie-user matrix
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        # create mapper from movie title to index
        hashmap = {
            movie: i for i, movie in enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)) # noqa
        }
        # transform matrix to scipy sparse matrix
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

        # clean up
        del df_movies, df_movies_cnt, df_users_cnt
        del df_ratings, df_ratings_filtered, movie_user_mat
        gc.collect()
        return movie_user_mat_sparse, hashmap

    def _fuzzy_matching(self, hashmap, fav_movie):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap,
                   fav_movie, n_recommendations):
        """
        return top n similar movie recommendations based on user's input movie
        Parameters
        ----------
        model: sklearn model, knn model
        data: movie-user matrix
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar movie recommendations
        """
        # fit : 전처리한 데이터 학습
        model.fit(data)
        # get input movie index
        print('You have input movie:', fav_movie)
        ## 입력받은 여기서 idx를 2개 받는다면..
        idx = self._fuzzy_matching(hashmap, fav_movie) # 입력된 movie와 맵핑된 index를 찾는다. 
        # inference
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)

        # get list of raw idx of recommendations
        raw_recommends = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1]
            )[:0:-1]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # return recommendation (movieId, distance)
        return raw_recommends

    def make_recommendations(self, fav_movie, n_recommendations):
        """
        make top n movie recommendations
        Parameters
        ----------
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        """
        # get data : 전처리 & 입력데이터 
        movie_user_mat_sparse, hashmap = self._prep_data()
        # get recommendations
        raw_recommends = self._inference(self.model, movie_user_mat_sparse, hashmap,
            fav_movie, n_recommendations)
        print('n_recommendations:', n_recommendations)
        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        # print('Recommendations for {}:'.format(fav_movie))

        # 제목(ran_title), 유사도 담기 
        total_title_=[]
        for i, (idx, dist) in enumerate(raw_recommends):
            # print('{0}: {1}'.format(i+1, reverse_hashmap[idx]))
            total_title_.append(reverse_hashmap[idx])

        # print('total_title_::', total_title_)
        return total_title_
            
        #     ran_title.append(reverse_hashmap[idx])
        #     ran_dist.append(dist)

        # for rt in range(0, len(ran_title)):
        #     print('ran_title: ', ran_title[rt])
        #     # print('ran_dist[rt]: ', ran_dist[rt])
        #     return ran_title[rt]
        #     # total_dist.append(ran_dist[rt])
        #     print('{0}: {1}, with distance .'
        #           'of {2}'.format(i+1, reverse_hashmap[idx], dist))
# 1: Kung Fu Panda (2008), with distance of 0.37368708848953247
# 2: Inception (2010), with distance of 0.3691744804382324



def parse_args():
    parser = argparse.ArgumentParser(
        prog="Movie Recommender",
        description="Run KNN Movie Recommender")
    # 데이터셋 경로 설정
    parser.add_argument('--path', nargs='?', default='../../dataset/',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--movie_name', nargs='?', default='',
                        help='provide your favoriate movie name')
    parser.add_argument('--top_n', type=int, default=5,
                        help='top n movie recommendations')
    return parser.parse_args()


#movie_name_ : 입력값
def main(movie_name_):
# if __name__ == '__main__':
    # get args
    # 문자열 그대로 가져감
    args = parse_args()
    # movie_name2 : 사용자 값이 들어감(복수개)
    movie_name_ = 'superman, nottinghills, avatar'
    movie_name2 = movie_name_.split(',') # 자체가 배열 
    # 나머지 입력값
    ratings_filename = args.ratings_filename #ratings.csv
    data_path = args.path # default path 찾아감
    movies_filename = args.movies_filename # movies.csv - str그대로다. 
    top_n = args.top_n 
    total_title = []
    ran = [] 
    # 입력값 차례로 knnmodel에 들어감
    for i in range(0, len(movie_name2)):
        movie_name = movie_name2[i] 
        # 1. initial recommender system
        recommender = KnnRecommender(
            os.path.join(data_path, movies_filename),
            os.path.join(data_path, ratings_filename))
       # <__main__.KnnRecommender object at 0x1a17f2d350>

        # 2. set params
        recommender.set_filter_params(50, 50) #user, movies의 안좋은 데이터 버림
        recommender.set_model_params(20, 'brute', 'cosine', -1) #  n_neighbors, algorithm, metric, n_jobs=None

        # 3. make recommendations
        total_title.append(recommender.make_recommendations(movie_name, top_n))
    # 확인
    # total_title = np.array(total_title)
    total_title = np.array(total_title)
    num = total_title.shape[0]*total_title.shape[1]
    total_title = total_title.reshape(num)
    
    import random
    # 랜덤 10개
    random.shuffle(total_title)
    for i in range(0, 10):
        ran.append(total_title[i])
    print('ran::', ran)
    # 함수 
    return ran
    
    # totla_title_fin = []
    # for i range(0, )
    
    
    # print('total_title:::', total_title)

# import csv

# import random
# for i in range(0, 3):
#     random.shuffle(total_title)
# rows = []
# for i in range(0, 10):
#     title = total_title[i]
#     rows.append({'num':i+1, 'title':title}) #dict형태
#     print(f'result : {i+1} {title}')

# with open("./recommendations.csv", 'w') as f :
#     fieldnames = ['num','title']
#     writer = csv.DictWriter(f, fieldnames = fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)


# f.close()


