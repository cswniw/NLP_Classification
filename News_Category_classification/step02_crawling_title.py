from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re, time

options = webdriver.ChromeOptions()
options.add_argument("lang=ko_KR")
options.add_argument("--no-sandbox")   #가상컴퓨터에서 실행할 때 이 코드 준다. ex)docusystem.
options.add_argument("--disable-dev-shm-usage")  # 리눅스 쓸 때
options.add_argument("disable-gpu")   # 리눅스 쓸 때

driver = webdriver.Chrome("./chromedriver", options=options)  # 뒤의 확장자exe빼고 준다.


# 크롤링 함수 선언
def crawl_title() :
    global titles    # 글로벌 선언 안해줘도 됨. 이경우.
    try:
        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{1}]/li[{0}]/dl/dt[2]/a'.format(i, j)).text
        # 기사 제목 텍스트
        title = re.compile("[^가-힇 ]").sub(" ", title)
        titles.append(title)
        print(title)
    except NoSuchElementException:      # 해당 xpath주소를 찾지 못하면 다른 xpath주소로 시도.
        title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[{1}]/li[{0}]/dl/dt/a'.format(i, j)).text
        title = re.compile("[^가-힇 ]").sub(" ", title)
        titles.append(title)
        print(title)
    except :        # NoSuchElementException 실패 시 'error'출력 후 패스.
        titles.append('error')
        pass



category = ["politics","Economic","Social", "Culture", "World", "IT"]       # 기사 카테고리 리스트
df_titles = pd.DataFrame()      # 저장할 데이터프레임 생성.

# pages = [344, 406, 596, 101, 131, 77]  # 각 카테고리 별로 페이지 수가 다르다. 너무 많으면 과학습되니 좀 줄여주자.
pages = [131, 131, 131, 101, 131, 77]   # 자료 갯수를 맞춰준다.

for l in range(6) :
    titles = []
    for k in range(1,pages[l]) :
        url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}".format(l, k)
        # l (101~106) : 카테고리 숫자 //  k는 카테고리 별 페이지 숫자

        driver.get(url)
        # time.sleep(0.05)   # 네트워크가 느릴 경우 쓰자. 느리면 페이지가 열리기 전에 코드 실행되서 에러남.
        # 또는 아래의 except Stale 코드 넣자.
        for j in range(1,5) :
            for i in range(1,6) :
                try :
                    crawl_title()
                except StaleElementReferenceException :
                    driver.get(url)
                    print("StaleElementReferenceException")   # 대신 이 오류 시 title 못 받는다.
                    time.sleep(1)    # 네트워크 문제 땜에 1초 이상 걸리면 코드가 죽을 수 있다.
                    crawl_title()
                except :
                    print("error")

        if k % 50 == 0 :    # 50번째 페이지 마다 저장.
            df_section_titles = pd.DataFrame(titles, columns=["title"])
            df_section_titles["category"] = category[l]
            df_section_titles.to_csv("./crawling/special_news_{}_{}-{}.csv".format(
                category[l], k-49, k), index=False)

            titles = []     # 저장했으면 중복방지를 위해 리스트를 비워준다.

    df_section_titles = pd.DataFrame(titles, columns=["title"])
    #  50번째 미만의 나머지 정보들도 저장하자.
    df_section_titles["category"] = category[l]
    df_section_titles.to_csv("./crawling/special_news_{}_remain.csv".format(category[l]), index=False) #index=False

    df_titles = pd.concat([df_titles, df_section_titles], axis="rows", ignore_index=True)
    # 6개의 카테고리 리스트에서 1개가 끝날 때마다 병합한다.

df_titles.to_csv("./crawling/special_naver_news.csv", index=False)
# 병합이 완료 후 csv 파일로 저장.

print(len(titles))

driver.close()
