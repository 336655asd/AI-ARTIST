from app.fstyle import magic
from app.fstyle import reshape

def style_interface(file_dir="no",style_iterm = 0):
    if file_dir=="no":
        print("nothing to do")
        return 0
    print("resize")
    file_name=reshape.resize(file_dir)
    print("start")
    magic.change_style(file_name,style_iterm)
    print("done")

if __name__ == "__main__":
    magic.change_style()
