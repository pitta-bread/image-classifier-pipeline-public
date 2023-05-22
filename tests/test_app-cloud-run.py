import requests
from configparser import ConfigParser

config = ConfigParser()
config.read(".config/config.ini")

url = config.get('tests', 'url')
my_img = {'image': open('test_bed.jpg', 'rb')}
r = requests.post(url, files=my_img)
print(r.text)
