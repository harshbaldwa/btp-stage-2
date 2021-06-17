import numpy as np
import requests
from bs4 import BeautifulSoup
import re

fo = open("spherical_points.py", "w")
fo.write("import numpy as np\n\ndef sph_points(n):\n")


url = "http://neilsloane.com/sphdesigns/dim3/"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

harsh = soup.find_all("a")[::4]
for a in harsh:
    result = re.search('des.3.(.*).*.txt', a["href"])
    data = result.group(1).split(".")
    new_url = url + a["href"]
    new_url = new_url.replace(" ", "")
    new_page = requests.get(new_url)
    txt = new_page.text.split("\n")[:-1]
    new_txt = ",\n\t\t\t".join([", ".join(txt[i:i+3]) for i in range(0,len(txt),3)])
    
    fo.write("\tif n == {}:\n".format(data[0]))
    fo.write("\t\tpoints = np.array([\n\t\t\t{}\n\t\t])".format(new_txt))
    fo.write("\n\t\torder = {}\n\n".format(data[1]))
    
    print(data)

fo.write("\treturn points, order\n")

fo.close()