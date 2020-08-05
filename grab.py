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
    




















