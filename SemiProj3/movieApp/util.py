import numpy as np
import pandas as pd
from django.conf import settings

def random_poster_urls(isIntro, num):
    if isIntro == True:
        movie_url_file_path = settings.MOVIE_50_URL_FILE_PATH
    else:
        movie_url_file_path = settings.MOVIES_FILE_PATH
    df = pd.read_csv(movie_url_file_path)
    df = df[['title', 'img_url']]
    sample_df = df.sample(n=num)
    res = sample_df.to_dict(orient="index")
    return res

def get_poster_urls(movie_list):
    movie_url_file_path = settings.MOVIES_FILE_PATH
    df = pd.read_csv(movie_url_file_path)
    new_df = pd.DataFrame(columns=['title', 'img_url'])
    for item in movie_list:
        new_df = new_df.append(pd.DataFrame(df[df['title'] == item], columns=['title', 'img_url']))
    res = new_df.to_dict(orient="index")
    return res

def get_poster_urls_to_list(movie_list):
    #print(movie_list)
    movie_url_file_path = settings.MOVIES_FILE_PATH
    df = pd.read_csv(movie_url_file_path)
    new_df = pd.DataFrame(columns=['title', 'img_url'])
    for item in movie_list:
        new_df = new_df.append(pd.DataFrame(df[df['title'] == item], columns=['title', 'img_url']))
    res = list(np.array(new_df['img_url'].tolist()))
    #print(res)
    return res

def get_naver_id_to_list(movie_list):
    print(movie_list)
    movie_url_file_path = settings.MOVIES_FILE_PATH
    df = pd.read_csv(movie_url_file_path)
    new_df = pd.DataFrame(columns=['title', 'naver_url'])
    for item in movie_list:
        new_df = new_df.append(pd.DataFrame(df[df['title'] == item], columns=['title', 'naver_url']))
    res = list(np.array(new_df['naver_url'].tolist()))
    print(res)
    return res

def get_name_from_dict(movie_dict):
    name_list = []
    for i in movie_dict.keys():
        a = list(movie_dict.get(i).values())
        name_list.append(a[0])
    return name_list

    
if __name__ == '__main__':
    # test1
    print("[teset 1]")
    tdict = random_poster_urls(5)
    print(tdict)

    print()
    # test2
    print("[teset 2]")
    tl = ["Mr. Holland's Opus (1995)",
          "It Takes Two (1995)",
          "Home for the Holidays (1995)",
          "Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)",
          "Mary Reilly (1996)"
    ]
    tdict2 = get_poster_urls(tl)
    print(tdict2)

    # test3
    print("[teset 3]")
    tlist = get_name_from_dict(tdict2)
    print(tlist)

    # test4
    print("[teset 4]")
    tlist2 = get_naver_id_to_list(tdict2)
    print(tlist2)

    # test5
    print("[teset 5]")
    tlist3 = get_poster_urls_to_list(tdict2)
    print(tlist3)