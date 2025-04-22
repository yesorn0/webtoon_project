import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
from webtoon_crawling_data import LezhinCrawler

crawler = LezhinCrawler()

# /html/body/div[2]/section/div[2]/div/button[2]
# /html/body/div[2]/section/div[2]/div/button[11]

categories = []
for i in range(2, 12):
    crawler.start_driver()
    time.sleep(1)
    category = crawler.driver.find_element(By.XPATH, '/html/body/div[2]/section/div[2]/div/button[{}]'.format(i))
    categories.append(category.text)
    category.click()
    time.sleep(2)
    crawler.get_links('//*[@id="exhibit-genreplus-comic-list"]/ul/li/a')
    crawler.crawl()

crawler.close()
crawler.categories = categories
crawler.save_to_csv('Lezhin')

# options = ChromeOptions()
# options.add_argument('lang=ko_KR')
# service = ChromeService(executable_path=ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service, options=options)
# driver.set_window_size(1920, 1080)
#
#
# df_classification = pd.DataFrame()
# plots = []
# categories = []
# titles = []
#
# url = 'https://www.lezhin.com/ko/genreplus?sub_tags=all&filter=all&order=popular'
# driver.get(url)
#
# time.sleep(1)
#
# # driver.find_element(By.XPATH, '//*[@id="gnb_login_button"]').click()
#
# # login_id_element = driver.find_element(By.XPATH , '//*[@id="id"]')
# # login_pw_element = driver.find_element(By.XPATH , '//*[@id="pw"]')
# # login_id_element.send_keys(login_id)
# # time.sleep(0.5)
# # login_pw_element.send_keys(login_pw)
# # time.sleep(0.5)
# # login_pw_element.send_keys(Keys.RETURN)
#
# # print(list(soup))
# # for i in range(5):
# #     driver.execute_script("window.scrollBy(0, 1000);")  # 세로 방향으로 1000px씩 내림
# #     time.sleep(1)
# for idx in range(2,12):
#     category_xpath = '/html/body/div[2]/section/div[2]/div/button[{}]'.format(idx)
#     try:
#         category_element = driver.find_element(By.XPATH, category_xpath)
#         category = category_element.text
#         category_element.click()
#         time.sleep(3)
#
#         wcd.bottom_to_scrolling(driver)
#
#         i = 1
#         while True:
#             webtoon_xpath = '//*[@id="exhibit-genreplus-comic-list"]/ul/li[{}]/a'.format(i)
#             title_xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/h2'
#             try:
#                 driver.find_element(By.XPATH, webtoon_xpath).click()
#
#                 time.sleep(1)  # 3. 페이지가 완전히 바뀔 때까지 잠깐 대기
#                 title = None
#                 plot = None
#
#                 while True:
#                     plot_xpath = '/html/body/div[2]/section/div/div[1]/div[2]/p'
#                     info_xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/button'
#
#                     try:
#                         driver.find_element(By.XPATH, info_xpath).click()
#                         time.sleep(0.5)
#                         title = driver.find_element(By.XPATH, title_xpath).text
#                         plot = driver.find_element(By.XPATH, plot_xpath).text
#                     except:
#                        print('error', i)
#                     if plot is not None: break
#                     time.sleep(1)
#
#                 titles.append(title)
#                 plots.append(plot)
#                 categories.append(category)
#
#                 # 4. 뒤로 가기 (브라우저의 뒤로가기 기능!)
#                 driver.back()
#                 # 5. 다시 돌아온 페이지를 위해 잠깐 대기
#                 time.sleep(1)
#
#                 i += 1      # i = i + 1
#             except:
#                 print('scrolling')
#                 driver.execute_script("window.scrollBy(0, 1000);")  # 세로 방향으로 1000px씩 내림
#                 time.sleep(1)
#                 new_height = driver.execute_script("return document.body.scrollHeight")
#                 if new_height == last_height:
#                     print(f"📌 카테고리 {category}에서 스크롤 종료")
#                     break
#                 last_height = new_height
#     except:
#         pass
#
# print(titles)
# print(plots)
# print(categories)
#
# df_classification['title'] = titles
# df_classification['plot'] = plots
# df_classification['category'] = categories
#
# df_classification.info()
# print(df_classification.head())
#
# df_classification.to_csv('./crawling_data/lezhin_category_plot.csv',index=False)