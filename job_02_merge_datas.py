import pandas as pd
from sklearn.utils import resample

kakao_data = pd.read_csv('./crawling_data/kakao_category_plot.csv')
naver_data = pd.read_csv('./crawling_data/naver_category_plot.csv')
lezhin_data = pd.read_csv('./crawling_data/Lezhin_webtoons.csv')

print(kakao_data.head())
print(naver_data.head())
print(lezhin_data.head())

print(kakao_data.category.unique())
print(naver_data.category.unique())
print(lezhin_data.category.unique())

categories = ['판타지', '드라마', '로맨스', '로판', '무협', '액션']

def extract_category(text):
    for category in categories:
        if category in text:
            return category
    return '기타'

# def clean_category(text):
#     text = re.sub(r'[\r\n]', '', text)
#     text = re.sub(r'탭.*', '', text)
#     text = re.sub(r'총\s*\d+개\s*중\s*\d+번째', '', text)
#     text = re.sub(r'#', '', text)
#     return text.strip()

kakao_data.category = kakao_data.category.apply(extract_category)
naver_data.category = naver_data.category.apply(extract_category)
lezhin_data.category = lezhin_data.category.apply(extract_category)

# '로판' 카테고리를 가진 행 제거
kakao_data = kakao_data[kakao_data.category != '로판']
# # '감성' 카테고리를 드라마로 변경
# naver_data.category = naver_data.category.replace({'감성' : '드라마'})

print(kakao_data.category.unique())
print(naver_data.category.unique())
print(lezhin_data.category.unique())

raw_data = pd.concat([kakao_data, naver_data, lezhin_data])

raw_data.info()
print(raw_data.category.unique())
print(raw_data.head())
print(raw_data.category.value_counts())

print(raw_data.columns)

# UnderSampling 데이터 손실
target_count = 4000
balanced_data = []
for category in raw_data['category'].unique():
    subset = raw_data[raw_data['category'] == category]
    if len(subset) > target_count:
        subset = resample(subset, replace=False, n_samples=target_count, random_state=42)
    else:
        subset = subset  # 그대로
    balanced_data.append(subset)

df_balanced = pd.concat(balanced_data)
print(df_balanced['category'].value_counts())


# OverSampling 과적합 주의
target_count = 3931
balanced_data = []
for category in df_balanced['category'].unique():
    subset = df_balanced[df_balanced['category'] == category]
    if len(subset) < target_count:
        subset = resample(subset, replace=True, n_samples=target_count, random_state=42)
    balanced_data.append(subset)

df_balanced = pd.concat(balanced_data)
print(df_balanced['category'].value_counts())
df_balanced = df_balanced[df_balanced.category != '기타']
df_balanced.to_csv('./crawling_data/webtoons_csv', index = False)