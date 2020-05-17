

def word2vecFn():
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from gensim.test.utils import common_texts
    from gensim.models.word2vec import Word2Vec

    # ========== 입력값 전처리 ========== #

    import os
    print("print : ", os.getcwd())

    data = pd.read_csv('data/movieList_users.csv')
    df_items = pd.read_csv('data/movies.csv')

    # 사용자한테 영화명을3개 받을것
    # 영화명이 들어오면 영화아이디로
    m_list = ['Toy Story (1995)','Jumanji (1995)','Grumpier Old Men (1995)']

    # m_list = [ ]

    # 숫자로 변환된 영화 아이디가 저장되는 리스트
    m_id_list = []
    # 숫자로 변환해 주기 위해 for문을 돌린다
    for m in m_list:
        m_id = df_items.loc[df_items['title'] == m]['movieId'].values
        m_id_list.append(m_id[0])


    # ========== 모델불러오기 ========== #
    # colab에서 movieList_users.csv를 통해 학습한 모델파일을 불러온다
    model = Word2Vec.load("data/word2vec.model")


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
    predic_data.to_csv('predic_data.csv', index=0)

    #     print(result_list[0][1])    
    result_title = predic_data[:5]['title']
    result_similar = predic_data[:5]['similar']
    result_similar = round((result_similar), 3)

    print(result_title, result_similar)

    return result_title, result_similar