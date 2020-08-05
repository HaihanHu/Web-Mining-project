from bs4 import BeautifulSoup
import requests
import json
import pandas as pd

class WebGrab(object):
    
    def __init__(self):
        self.df = pd.DataFrame()
        self.json_data = None
        
    def writeToFile(self):
        self.df.to_csv('features.csv')
        
    def getSingleKey(self, key):
        value = ""
        if key in self.json_data:
            value = self.json_data[key]
        return value
    
    def getSecondKey(self, key1, key2):
        value = ""
        if key1 in self.json_data:
            if key2 in self.json_data[key1]:
                value = self.json_data[key1][key2]
        return value
        
    def getProjectJson(self, pageNum):
        
        ids = []
        names = []
        blurbs = []
        goals = []
        pledgeds = []
        currencys = []
        deadlines = []
        create_dates = []
        launched_dates = []
        backers_counts = []
        percents = []
        
        authors = []
        loc_citys = []
        loc_states = []
        loc_countrys = []
        cate_names = []
        cate_slugs = []
        
        urls = []
    
        for i in range(pageNum):
            
            print('page'+str(i))
            url = 'https://www.kickstarter.com/discover/advanced?sort=end_date&seed=2567694&page='+str(i)
            
            response=requests.get(url,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
            html=response.content
            
            soup = BeautifulSoup(html.decode('ascii', 'ignore'),'html.parser')
            modules = soup.findAll('div',{'class':'js-react-proj-card col-full col-sm-12-24 col-lg-8-24'})
            for module in modules:
                
                jsonStr = module['data-project']
                self.json_data = json.loads(jsonStr)
                
                ID = self.getSingleKey("id")
                name = self.getSingleKey("name")
                blurb = self.getSingleKey("blurb")
                goal = self.getSingleKey("goal")
                pledged = self.getSingleKey("pledged")
                currency = self.getSingleKey("currency")
                deadline = self.getSingleKey("deadline")
                created_at = self.getSingleKey("created_at")
                launched_at = self.getSingleKey("launched_at")
                backers_count = self.getSingleKey("backers_count")
                percentage = self.getSingleKey("percent_funded")
                
                author = self.getSecondKey("creator", "name")
                loc_city = self.getSecondKey("location", "name")
                loc_state = self.getSecondKey("location", "state")
                loc_country = self.getSecondKey("location", "country")
                cate_name = self.getSecondKey("category", "name")
                cate_slug = self.getSecondKey("category", "slug")
                
                url = self.json_data['urls']['web']['project']
                if len(url) == 0:
                    url = 'None'
                    
                ids.append(ID)
                names.append(name)
                blurbs.append(blurb)
                goals.append(goal)
                pledgeds.append(pledged)
                currencys.append(currency)
                deadlines.append(deadline)
                create_dates.append(created_at)
                launched_dates.append(launched_at)
                backers_counts.append(backers_count)
                percents.append(percentage)
                
                authors.append(author)
                loc_citys.append(loc_city)
                loc_states.append(loc_state)
                loc_countrys.append(loc_country)
                cate_names.append(cate_name)
                cate_slugs.append(cate_slug)
                urls.append(url)
                
        self.df["ID"] = ids
        self.df["pro_name"] = names
        self.df["blurb"] = blurbs
        self.df["goal"] = goals
        self.df["pledged"] = pledgeds
        self.df["currency"] = currencys
        self.df["deadline"] = deadlines
        self.df["created_at"] = create_dates
        self.df["launched_at"] = launched_dates
        self.df["backers_count"] = backers_counts
        self.df["percent_funded"] = percents
        self.df["author"] = authors
        self.df["loc_city"] = loc_citys
        self.df["loc_state"] = loc_states
        self.df["loc_country"] = loc_countrys
        self.df["cate_name"] = cate_names
        self.df["cate_slug"] = cate_slugs
        self.df["url"] = urls
        
        self.writeToFile()
                
        return 0
    

        
        


































