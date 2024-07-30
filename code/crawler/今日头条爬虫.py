import os
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import ahocorasick

class Toutiao:
    def __init__(self,keywords):

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
        chrome_options.add_argument("--disable-blink-features")
        chrome_options.add_argument("--headless")  # 启用无头模式
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0')
        s = Service('./chromedriver.exe')
        self.browser = webdriver.Chrome(service=s, options=chrome_options)

        self.keywords = keywords
        self.tag_tree = ahocorasick.Automaton()
        for word in keywords:
            self.tag_tree.add_word(word,word)
        self.tag_tree.make_automaton()

    def search(self, pageNumber=10):
        for page in range(pageNumber):
            self.browser.get('https://so.toutiao.com/search?dvpf=pc&source=input&keyword={}&pd=information&action_type=search_subtab_switch&page_num={}&from=news&cur_tab_title=news'.format("".join(self.keywords), page))
            time.sleep(3)
            # print(self.browser.page_source)
            cards = self.browser.find_elements(By.CLASS_NAME, 'cs-view.cs-view-block.cs-card-content')
            urls = []
            introductions = []
            for card in cards:
                url = card.find_element(By.CLASS_NAME, 'text-ellipsis.text-underline-hover').get_attribute("href")
                urls.append(url)
                print(url)
                introduction = "-"
                try:
                    introduction = card.find_elements(By.CLASS_NAME, "cs-view.cs-view-block.cs-text.align-items-center")[1].find_element(By.CLASS_NAME, "text-underline-hover").text
                except:
                    pass
                introductions.append(introduction)
                # print(introduction)

            for i in range(len(urls)):

                self.browser.get(urls[i])
                time.sleep(3)

                try:
                    text = self.browser.find_element(By.CLASS_NAME, "article-content").text
                except:
                    print("")

                r = list(self.tag_tree.iter(text))
                if len(set([i[1] for i in r])) == len(self.keywords):

                    files = os.listdir("result")
                    with open(f"./result/{len(files)}.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                else:
                    print("不相干")


if __name__ == "__main__":
    Toutiao(["小米汽车"]).search()