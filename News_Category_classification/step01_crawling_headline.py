from bs4 import BeautifulSoup
import requests, re, datetime
import pandas as pd

pd.set_option("display.unicode.east_asian_width", True)

print(datetime.datetime.today().strftime("%Y%m%d"))
# 현재 시각

url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100"
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"}
resp = requests.get(url, headers=headers)

# print(list(resp))     # html 코드 가져옴.
# print(type(resp))     # <class 'requests.models.Response'>

soup = BeautifulSoup(resp.text, "html.parser")
# print(soup)

title_tags = soup.select(".cluster_text_headline")      # 찾고자하는 클래스를 선택
# print(title_tags)
# print(type(title_tags[0]))

titles = []
for title_tag in title_tags :
    titles.append(re.compile("[^가-힇|a-z|A-Z ]").sub(" ", title_tag.text))
                            # 정규표현식 //   ^ 다음에 오는 것들을 제외하고 그자리를 
                            # 빈칸으로 채워라 어디서? title_tag에서. // 0-9 가능하나. 이번에는 빼자 데이터 처리를 위해.

# print(titles)
# print(len(titles))
df_titles = pd.DataFrame()   # 빈 데이터 프레임 만들기.
re_title = re.compile("[^가-힇|a-z|A-Z ]")

category = ["Politics","Economic","Social","Culture","World", "IT"]


for i in range(6) :
    headers_2 = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"}
    resp_2 = requests.get("https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}".format(i), headers=headers_2)
    soup_2 = BeautifulSoup(resp_2.text, "html.parser")
    title_tags_2 = soup_2.select(".cluster_text_headline")
    
    titles_2 = []

    for title_tag_2 in title_tags_2 :
        titles_2.append(re.compile("[^가-힇|a-z|A-Z ]").sub(" ", title_tag_2.text))
        # 기사 제목 크롤링
    
    df_section_titles = pd.DataFrame(titles_2, columns=["title"])       # 데이터 프레임화
    df_section_titles["category"] = category[i]     # 카테고리에 해당 기사의 섹션
    df_titles = pd.concat([df_titles, df_section_titles], axis="rows", ignore_index=True)
    # 디폴트는 axis="rows"
    # 카테고리 6개 각 기사 제목 크롤링

    print("https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10%d" %(i))\

print(len(titles_2))

print(df_titles.head())
print(df_titles.info())
print(df_titles["category"].value_counts())

print(titles[0])
print(titles[-1])

df_titles.to_csv("./crawling/special_naver_headline_news_{}.csv".format(
    datetime.datetime.today().strftime("%Y%m%d")), index=False)
