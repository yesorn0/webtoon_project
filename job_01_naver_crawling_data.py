import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
from webtoon_crawling_data import NaverCrawler

crawler = NaverCrawler()

crawler.start_driver()
crawler.get_links('//*[@id="content"]/div[1]/ul/li/a')
titles, plots, categories = crawler.crawl()
crawler.save_to_csv('naver')
crawler.close()

print(titles)
print(plots)
print(categories)

# options = ChromeOptions()
#
# options.add_argument('lang=ko_KR')
# service = ChromeService(executable_path=ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service, options=options)
#
# df_classification = pd.DataFrame()
# plots, categories, titles = [], [], []
#
# url = 'https://comic.naver.com/webtoon?tab=finish'
# driver.get(url)
#
# time.sleep(1)
#
# last_height = driver.execute_script("return document.body.scrollHeight")
# while True:
#     # print('scrolling')
#     driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")  # 세로로 스크롤 높이만큼 내림
#     time.sleep(1)
#     new_height = driver.execute_script("return document.body.scrollHeight")
#     if new_height == last_height:
#         break
#     last_height = new_height
#
# cnt = 0
# title_xpath = '//*[@id="content"]/div[1]/div/h2'
# plot_xpath = '//*[@id="content"]/div[1]/div/div[2]/p'
# category_xpath = '//*[@id="content"]/div[1]/div/div[2]/div/div/a[1]'
# hrefs = driver.find_elements(By.XPATH, '//*[@id="content"]/div[1]/ul/li/a')
# links = [link.get_attribute('href') for link in hrefs]
#
# total = len(hrefs)
#
# # links = []
# # for link in hrefs:
# #     links.append(link.get_attribute('href'))
#
# # links = []
# # for (int i = 0; i < sizeof(hrefs); i++)
# # {
# #     links.append(hrefs[i].get_attribute('href'))
# # }
#
# for link in links:
#     driver.get(link)
#     time.sleep(2)
#     try:
#         title = driver.find_element(By.XPATH, title_xpath).text
#         plot = driver.find_element(By.XPATH, plot_xpath).text
#         category = driver.find_element(By.XPATH, category_xpath).text
#
#         titles.append(title)
#         plots.append(plot)
#         categories.append(category)
#
#     except:
#         cnt += 1
#         print(f'Error Count : {cnt}')
#
# print(len(titles), len(plots), len(categories))
#
# print(titles)
# print(plots)
# print(categories)
#
# df_classification['title'] = titles
# df_classification['plot'] = plots
# df_classification['category'] = categories
# print(f'{len(df_classification) - cnt} / {total}')
# df_classification.info()
# print(df_classification.head())
#
# df_classification.to_csv('./crawling_data/webtoon_category_plot.csv',index=False)
#
# 2481개 중 2295번까지 완료... 77개의 에러