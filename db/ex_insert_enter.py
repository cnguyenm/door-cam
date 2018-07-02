from tables import GoIn, Person
from datetime import date
import time

# start counting
t1 = time.time()

# given a name
name = "Chau3"

# get the person with name
# if cannot find, create them
p1, created = Person.get_or_create(name=name)
print(p1)

# get entry GoIn of that person in today
today = date.today()
g1 = GoIn.select().where(
    (GoIn.person == p1) &
    (GoIn.date_enter.year == today.year) &
    (GoIn.date_enter.month == today.month) &
    (GoIn.date_enter.day == today.day)
)

# if found, get it
# default date=today, n_enter=0
goEntry = None
if g1.count() > 0:
    print("[db] get GoIn entry")
    goEntry = g1.get()
else:
    print("[db] create new GoIn entry")
    goEntry = GoIn.create(person=p1)

# update n_enter
goEntry.n_enter = goEntry.n_enter + 1
goEntry.save()
print("[db] update entry")
print("[db] new entry: " + str(goEntry.n_enter))

# print time
t2 = time.time()
time_elapse = t2 - t1
print("time: {}".format(time_elapse))



