from app.poem import sample
from app.darknet import darknet
from app.mix import text_cv

def poem_interface(poem_start):
    sample.poem_genetate(poem_start)

def darknet_interface():
    darknet.sth_generate()

def mix_interface():
    text_cv.mix()

if __name__ == "__main__":
    poem_genetate()
