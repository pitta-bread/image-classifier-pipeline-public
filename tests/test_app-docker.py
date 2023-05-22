import requests

url = 'http://localhost:8080/'
my_img = {'image': open('test_bed.jpg', 'rb')}
r = requests.post(url, files=my_img)
print(r.text)
