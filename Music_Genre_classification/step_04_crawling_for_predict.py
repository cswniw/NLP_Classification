from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re, time, random

### 학습한 모델을 테스트할 데이터 수집

### 크롬 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument("lang=ko_KR")
options.add_argument("--no-sandbox")   #가상컴퓨터에서 실행할 때 이 코드 준다. ex)docusystem.
options.add_argument("--disable-dev-shm-usage")  # 리눅스 쓸 때
options.add_argument("disable-gpu")   # 리눅스 쓸 때
driver = webdriver.Chrome("./chromedriver", options=options)  # 뒤의 확장자exe빼고 준다.


df_lyrics_total = pd.DataFrame()

category = ["Ballad", "Dance", "Rap_Hiphop", "RnB_Soul", "indi", "Rock_Metal", "Trot", "Folk_Blues"]

for k in range(100,801,100) :
        lyrics = []
        if k == 500 : continue
        else :
                ran_page = random.randint(13000, 14500)
                j = ran_page

                url = 'https://www.melon.com/genre/song_list.htm?gnrCode=GN0{}#params%5BgnrCode%5D=GN0{}' \
                      '&params%5BdtlGnrCode%5D=&params%5BorderBy%5D=NEW&params%5BsteadyYn%5D=' \
                      'N&po=pageObj&startIndex={}'.format(k,k,j)
                driver.get(url)
                time.sleep(1)

                # numbers = []
                # ran_num = random.randint(1, 51)
                # for i in range(10):
                #         while ran_num in numbers:
                #                 ran_num = random.randint(1, 51)
                #         numbers.append(ran_num)
                #
                # numbers.sort()

                # for i in numbers :
                for i in range(1,51) :
                        try:
                                driver.find_element_by_xpath('//*[@id="frm"]/div/table/tbody/tr[{}]/td[4]/div/a'
                                                             .format(i)).click()
                                lyric = driver.find_element_by_class_name('lyric').text
                                lyric = re.compile("[^가-힇a-zA-Z ]").sub(" ", lyric)
                                lyrics.append(lyric)
                                # print(lyric)
                                driver.back()
                        except NoSuchElementException:
                                print("None")
                                driver.back()

                        except StaleElementReferenceException:
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

        df_lyrics = pd.DataFrame(lyrics, columns=["lyric"])
        df_lyrics["category"] = category[(int((k-100)/100))]
        df_lyrics_total = pd.concat([df_lyrics_total, df_lyrics], axis="rows", ignore_index=True)


df_lyrics_total.to_csv("./data_for_predict/category_lyrics_total.csv")

driver.close()






