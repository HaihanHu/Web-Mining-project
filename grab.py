'''
Team 4
2018-10-26
'''

import re
import requests
import _thread 
import time
from bs4 import BeautifulSoup
from WebGrab import WebGrab
from ContentGrab import ContentGrab
import pandas as pd

def get(fileName, rang):
   grab = ContentGrab(fileName)
   grab.getProjectContent(rang)
   
if __name__ == '__main__':
    
    _thread.start_new_thread(get,('pro_contents.csv', range(800)))
    _thread.start_new_thread(get,('pro_contents2.csv', range(800, 1600)))
    _thread.start_new_thread(get,('pro_contents3.csv', range(1600, 2400)))

    grab = WebGrab()
    grab.getProjectJson(201)
    
    grab = ContentGrab('pro_contents.csv')
    grab2 = ContentGrab('pro_contents2.csv')
    grab3 = ContentGrab('pro_contents3.csv')
    _thread.start_new_thread(grab.getProjectContent(range(1000)))
    _thread.start_new_thread(grab2.getProjectContent(range(1000, 2000)))
    _thread.start_new_thread(grab3.getProjectContent(range(2000, 2400)))
    
    
#    data = pd.read_csv('features.csv')
#    urls = data.url
#    
#    url = urls[1336]
#    print(url)
#            
#    response=requests.get(url,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
#    html=response.content
#    
#    soup = BeautifulSoup(html.decode('ascii', 'ignore'),'html.parser')
#    
#    title = ""
#    content = ""
#    imageCount = 0
#    countVideo = 0
#    review = soup.find('div', {'class':re.compile('full-description')})
#    if review:
#        txts = review.findAll('p')
#        for txt in txts:
#            content += txt.text
#            
#        assets = review.findAll('div', {'class':"template asset"})
#        imageCount = len(assets)
#    
#    top = soup.find('div', {'class':re.compile('aspect-ratio')})
#    if top:
#        video = top.find('video')
#        countVideo = 1 if video else 0
#    
#    titleNode = soup.find('meta', {'property':"og:title"})
#    if titleNode:
#        if 'content' in titleNode.attrs:
#            title = titleNode['content']
#    print(title['content'])

































