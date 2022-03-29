import pandas as pd
import glob

data_paths = glob.glob("./crawling/special_crawling/*.csv")
# print(data_paths)
# 해당 폴더의 csv파일 리스트

df = pd.DataFrame()

for data_path in data_paths :
    df_temp = pd.read_csv(data_path)
    df = pd.concat([df, df_temp])
# title 카테고리에 Nan 값이 12개 있다.

df.dropna(inplace=True)
# Nan값 drop
df.reset_index(drop=True, inplace=True)
# drop 후 인덱스 재부여

df.to_csv("./crawling/special_crawling/total_naver_news.csv", index=False)

print(df.head())
print(df.tail())
print(df["category"].value_counts())
print(df.info())

