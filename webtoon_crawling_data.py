import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time

class BaseCrawler:
    # 공통 기능 구현 (스크롤링, 드라이버 초기화, 링크 수집 등)
    def __init__(self, url):
        self.driver = self.init_driver()
        self.url = url
        self.links = None
        self.titles = []
        self.plots = []
        self.categories = []

    def start_driver(self):
        self.driver.get(self.url)

    def init_driver(self):
        options = ChromeOptions()
        options.add_argument('--start-maximized')  #브라우저가 최대화된 상태로 실행됩니다.
        options.add_argument('--blink-settings=imagesEnabled=false')  # 브라우저에서 이미지 로딩을 하지 않습니다.
        options.add_argument('incognito')  # 시크릿 모드의 브라우저가 실행됩니다.
        options.add_argument('lang=ko_KR')
        service = ChromeService(executable_path=ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def scroll_to_bottom(self, wait_time=2):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(wait_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def get_links(self, xpath):
        self.scroll_to_bottom()
        elements = self.driver.find_elements(By.XPATH, xpath)
        self.links = [el.get_attribute("href") for el in elements]
        return self.links

    def get_text(self, xpath):
        text = self.driver.find_element(By.XPATH, xpath).text
        return text

    def save_to_csv(self, platform_name):
        filepath = f"./crawling_data/{platform_name}_webtoons.csv"
        df = pd.DataFrame({
            'title': self.titles,
            'plot': self.plots,
            'category': self.categories
        })
        df.to_csv(filepath, index=False)

    def close(self):
        self.driver.quit()

class NaverCrawler(BaseCrawler):
    def __init__(self):
        url = "https://comic.naver.com/webtoon?tab=finish"
        super().__init__(url)
        self.title_xpath = '//*[@id="content"]/div[1]/div/h2'
        self.plot_xpath = '//*[@id="content"]/div[1]/div/div[2]/p'
        self.category_xpath = '//*[@id="content"]/div[1]/div/div[2]/div/div/a[1]'

    def crawl(self):
        if self.links is None : return
        cnt = 0
        for link in self.links:
            self.driver.get(link)
            time.sleep(1)
            try :
                self.titles.append(self.get_text(self.title_xpath))
                self.plots.append(self.get_text(self.plot_xpath))
                self.categories.append(self.get_text(self.category_xpath))
            except :
                cnt += 1
                print(f"not datas,{cnt}")
                pass

        return self.titles, self.plots, self.categories

class KakaoCrawler(BaseCrawler):
    def __init__(self):
        url = 'https://page.kakao.com/menu/10010/screen/82?subcategory_uid=0'
        super().__init__(url)
        self.title_xpath = '//*[@id="__next"]/div/div[2]/div[1]/div/div[1]/div[1]/div/div[2]/a/div/span[1]'
        self.plot1_xpath = '//*[@id="__next"]/div/div[2]/div[1]/div/div[2]/div[2]/div/div/div[1]/div/div[2]/div/div'
        self.plot2_xpath = '//*[@id="__next"]/div/div[2]/div[1]/div/div[2]/div[2]/div/div/div[1]/div/div[2]/div[2]/div'
        self.info_xpath = '//*[@id="__next"]/div/div[2]/div[1]/div/div[2]/div[1]/div/div/div[2]/a'

    def crawl(self, category):
        if self.links is None: return
        cnt = 0
        for link in self.links:
            link = link + '?tab_type=about'
            self.driver.get(link)
            time.sleep(1)
            try :
                self.titles.append(self.get_text(self.title_xpath))
                self.plots.append(self.get_text(self.plot1_xpath))
                self.categories.append(category)
            except :
                try:
                    self.plots.append(self.get_text(self.plot2_xpath))
                except:
                    cnt += 1
                    print(f"not datas,{cnt}")
                    pass
                cnt += 1
                print(f"not datas,{cnt}")
                pass
        return self.titles, self.plots, self.categories

class LezhinCrawler(BaseCrawler):
    def __init__(self):
        url = "https://www.lezhin.com/ko/genreplus?sub_tags=all&filter=all&order=popular"
        super().__init__(url)
        self.title_xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/h2'
        self.plot_xpath = '/html/body/div[2]/section/div/div[1]/div[2]/p'
        self.info_xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/button'

    def crawl(self, category):
        if self.links is None: return
        cnt = 0
        for link in self.links:
            self.driver.get(link)
            time.sleep(1)
            try :
                self.driver.find_element(By.XPATH, self.info_xpath).click()
                time.sleep(1)
            except:
                pass
            try :
                self.titles.append(self.get_text(self.title_xpath))
                self.plots.append(self.get_text(self.plot_xpath))
                self.categories.append(category)
            except :
                cnt += 1
                print(f"not datas,{cnt}")
                pass
        return self.titles, self.plots, self.categories