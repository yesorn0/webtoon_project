from webtoon_crawling_data import NaverCrawler

crawler = NaverCrawler()

crawler.url = 'https://comic.naver.com/webtoon?tab=mon'

crawler.start_driver()
crawler.get_links('//*[@id="content"]/div[1]/ul/li/a')
titles, plots, categories = crawler.crawl()
crawler.save_to_csv('naver_predict')

print(titles)
print(plots)
print(categories)