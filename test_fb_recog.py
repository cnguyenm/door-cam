from pprint import pprint
from fbrecog import FBRecog
import time
import config

# useful variable
path = "img/new2.jpg"

# token fb graph API v2.12
access_token = config.access_token
cookie       = config.cookie
fb_dtsg      = config.fb_dtsg


# instantiate the recog class
t1 = time.time()

# recog can be
# [{'name': 'Nguyễn Minh Châu', 'certainity': 0.99593985080719}]
# or [] empty
# take about 5 seconds
recog = FBRecog(access_token, cookie, fb_dtsg)

print(recog.recognize(path))
t2 = time.time()
print("time: " + str(t2 - t1))

