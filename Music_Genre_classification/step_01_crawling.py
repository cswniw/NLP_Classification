from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re, time, os

### 크롬드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument("lang=ko_KR")
options.add_argument("--no-sandbox")   #가상컴퓨터에서 실행할 때 이 코드 준다. ex)docusystem.
options.add_argument("--disable-dev-shm-usage")  # 리눅스 쓸 때
options.add_argument("disable-gpu")   # 리눅스 쓸 때
driver = webdriver.Chrome("./chromedriver", options=options)  # 뒤의 확장자exe빼고 준다.

### 카테고리 리스트
category = ["Ballad", "Dance", "Rap_Hiphop", "RnB_Soul", "indi", "Rock_Metal", "Trot", "Folk_Blues"]


for k in range(100,801,100) :
        lyrics = []
        if k == 500 : continue      ### 숫자 500을 포함한 웹페이지 주소는 skip. (수집 대상이 아님.)
        else :
            for j in range(1, 10002, 50):   ### j는 페이지 수
                url = 'https://www.melon.com/genre/song_list.htm?gnrCode=GN0{}#params%5BgnrCode%5D=GN0{}' \
                      '&params%5BdtlGnrCode%5D=&params%5BorderBy%5D=NEW&params%5BsteadyYn%5D=' \
                      'N&po=pageObj&startIndex={}'.format(k,k,j)
                driver.get(url)
                time.sleep(1)   ### 웹사이트로부터 ddos공격으로 인식되어 차단되지 않게 타임슬립.
                for i in range(1,51) :     # 웹페이지 하나 당 50곡 리스트에 접근
                        try:
                                driver.find_element_by_xpath('//*[@id="frm"]/div/table/tbody/tr[{}]/td[4]/div/a'
                                                             .format(i)).click()
                                lyric = driver.find_element_by_class_name('lyric').text
                                lyric = re.compile("[^가-힇a-zA-Z ]").sub(" ", lyric)     # 문자 데이터 수집
                                lyrics.append(lyric)
                                # print(lyric)
                                driver.back()
                        except NoSuchElementException:  # 해당 html 요소가 없을 경우 예외처리.
                                print("None")
                                driver.back()

                        except StaleElementReferenceException:  # 로딩이 지연될 시.
                                time.sleep(1)
                                try:
                                        driver.find_element_by_xpath(
                                                '//*[@id="frm"]/div/table/tbody/tr[{}]/td[4]/div/a'
                                                        .format(i)).click()
                                        lyric = driver.find_element_by_class_name('lyric').text
                                        lyric = re.compile("[^가-힇a-zA-Z ]").sub(" ", lyric)
                                        lyrics.append(lyric)
                                        # print(lyric)
                                        driver.back()
                                except NoSuchElementException:
                                        driver.back()

                if j % 500 == 1 :   # 500곡의 가사를 수집할 때 마다 저장.
                        df_lyrics = pd.DataFrame(lyrics, columns=["lyric"])
                        df_lyrics["category"] = category[(int((k - 100) / 100))]
                        df_lyrics.to_csv("./crawling/{}_lyrics_{}-{}.csv".
                                         format(category[(int((k - 100) / 100))], j - 500, j), index=False)
                        lyrics = []

        df_lyrics = pd.DataFrame(lyrics, columns=["lyric"])     # 500곡 미만의 가사를 저장.
        df_lyrics["category"] = category[(int((k - 100) / 100))]
        df_lyrics.to_csv("./crawling/{}_lyrics_remain.csv".
                         format(category[(int((k - 100) / 100))]), index=False)

driver.close()


def merge():    # 500곡 단위마다 수집된 데이터를 통합하기 위한 함수.
    working_dir = ''.join([os.getcwd(), '/crawling']).replace('\\', '/')
    file_list = os.listdir(working_dir)
    result = []

    for file_name in file_list:
        result.append(file_name)

    _merge_data = pd.DataFrame()
    for file_name in result:
        csv_file = pd.read_csv(f'./crawling/{file_name}')
        _merge_data = pd.concat([_merge_data, csv_file], axis='rows', ignore_index=False)

    _merge_data.reset_index(drop=True, inplace=True)
    _merge_data.to_csv(f'./crawling/category_lyrics_final.csv')
    print(f'Success to save "final.csv"')

merge()     # 함수 실행.


















