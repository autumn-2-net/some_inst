import PIL
from PIL import ImageFont, Image, ImageDraw

# 生成font
ttf_path = 'Aa剑豪体.ttf'
text_size = 100 # text_size 是字号
font = ImageFont.truetype(ttf_path, text_size)

x,y =font.getsize('我')
print(x,'--',y)