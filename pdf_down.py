import requests

url = 'https://res.cloudinary.com/ddeu2euos/image/upload/v1710517017/ppnfxzzis2euvm8un40j.pdf'
response = requests.get(url)

with open('sample.pdf', 'wb') as f:
    f.write(response.content)