import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time

options = ChromeOptions()

options.add_argument('lang=ko_KR')
service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# with open("naver_login.txt", "r") as file:
#     lines = file.readlines()
#     login_id = lines[0].strip()
#     login_pw = lines[1].strip()

Crawling_Data_Max = 3000

df_classification = pd.DataFrame()
plots = []
categories = []
titles = []

url = 'https://comic.naver.com/webtoon?tab=finish'
driver.get(url)

time.sleep(1)

# driver.find_element(By.XPATH, '//*[@id="gnb_login_button"]').click()

# login_id_element = driver.find_element(By.XPATH , '//*[@id="id"]')
# login_pw_element = driver.find_element(By.XPATH , '//*[@id="pw"]')
# login_id_element.send_keys(login_id)
# time.sleep(0.5)
# login_pw_element.send_keys(login_pw)
# time.sleep(0.5)
# login_pw_element.send_keys(Keys.RETURN)

# print(list(soup))
# for i in range(5):
#     driver.execute_script("window.scrollBy(0, 1000);")  # 세로 방향으로 1000px씩 내림
#     time.sleep(1)

for i in range(1, Crawling_Data_Max+1):
    webtoon_xpath = '//*[@id="content"]/div[1]/ul/li[{}]/a'.format(i)
    title_xpath = '//*[@id="content"]/div[1]/div/h2'
    plot_xpath = '//*[@id="content"]/div[1]/div/div[2]/p'
    category_xpath = '//*[@id="content"]/div[1]/div/div[2]/div/div/a[1]'

    try:
        driver.find_element(By.XPATH, webtoon_xpath).click()

        time.sleep(1)  # 3. 페이지가 완전히 바뀔 때까지 잠깐 대기

        try:
            title = driver.find_element(By.XPATH, title_xpath).text
            plot = driver.find_element(By.XPATH, plot_xpath).text
            category = driver.find_element(By.XPATH, category_xpath).text

            titles.append(title)
            plots.append(plot)
            categories.append(category)
        except:
            print('error', i)

        # 4. 뒤로 가기 (브라우저의 뒤로가기 기능!)
        driver.back()
        # 5. 다시 돌아온 페이지를 위해 잠깐 대기
        time.sleep(1)

    except:
        print('scrolling')
        driver.execute_script("window.scrollBy(0, 1000);")  # 세로 방향으로 1000px씩 내림
        time.sleep(1)

print(titles)
print(plots)
print(categories)

df_classification['title'] = titles
df_classification['plot'] = plots
df_classification['category'] = categories

df_classification.info()
print(df_classification.head())

# # df_classification.to_csv('./crawling_data/webtoon_category_plot.csv',index=False)

# for i in range(5):
#     time.sleep(0.5)
#
# time.sleep(5)
#
#
# for i in range(1, 5):
#     for j in range(1, 7):
#         title_path = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i, j)
#         try:
#             title = driver.find_element(By.XPATH, title_path).text
#             print(title)
#         except:
#             print('error', i, j)

