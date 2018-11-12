#-*- coding: utf-8 -*-
import cv2
import ch_text
import codecs

def mix():
    
    img = cv2.imread('app/static/art/res.jpg')

    color = (255, 255, 255)  # Green
    pos1 = (10, 10)
    pos2 = (10,45)
    text_size = 30
    f=codecs.open('app/poem.txt','r','utf-8')
    text=f.read()
    f.close()
    lines=text.split(u'。')
    line1="".join(lines[0])
    line2="".join(lines[1])
    # ft = put_chinese_text('wqy-zenhei.ttc')
    ft = ch_text.put_chinese_text('app/mix/hwxk.ttf')
    image = ft.draw_text(img, pos1, line1, text_size, color)
    image = ft.draw_text(image, pos2, line2, text_size, color)
    cv2.imwrite('app/static/art/res_poem.jpg',image)
