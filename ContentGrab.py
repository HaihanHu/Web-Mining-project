'''
Team 4
2018-11-18
'''
import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

class ContentGrab(object):
    
    def __init__(self, fileName):
        self.df = pd.DataFrame()
        self.fileName = fileName
        self.json_data = None
        
    def writeToFile(self):
        self.df.to_csv(self.fileName)
        
    def getProjectContent(self, rang):
        
        data = pd.read_csv('features.csv')
        urls = data.url
        
        names = []
        videoCounts = []
        imageCounts = []
        contents = []
        
        for i in rang:
            
            url = urls[i]
            
            response=requests.get(url,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
            html=response.content
            
            soup = BeautifulSoup(html.decode('ascii', 'ignore'),'html.parser')
            
            title = ""
            content = ""
            imageCount = 0
            countVideo = 0
            review = soup.find('div', {'class':re.compile('full-description')})
            if review:
                txts = review.findAll('p')
                for txt in txts:
                    content += txt.text
                    
                assets = review.findAll('div', {'class':"template asset"})
                imageCount = len(assets)
            
            top = soup.find('div', {'class':re.compile('aspect-ratio')})
            if top:
                video = top.find('video')
                countVideo = 1 if video else 0
            
            titleNode = soup.find('meta', {'property':"og:title"})
            if titleNode:
                if 'content' in titleNode.attrs:
                    title = titleNode['content']
            
            names.append(title)
            videoCounts.append(countVideo)
            imageCounts.append(imageCount)
            contents.append(content)
            
            print(i)

        self.df["name"] = names
        self.df["videoCount"] = videoCounts
        self.df["imageCount"] = imageCounts
        self.df["content"] = contents

        self.writeToFile()
        return 0


























