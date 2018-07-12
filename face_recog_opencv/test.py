
import datetime

def test1():
    try:
        with open('file_fake.picle', 'rb') as fp:
            a = fp.readline()
    except FileNotFoundError:
        print("file not found")


def test2():
    now = datetime.datetime.now()
    print(now)

if __name__=="__main__":
    test2()



