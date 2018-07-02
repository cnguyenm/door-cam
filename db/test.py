import peewee
import datetime
from tables import Person, GoIn

db = peewee.SqliteDatabase("wee.db")

# persons = Person.select()
# persons_list = list(persons)
# print(persons_list)

enters = GoIn.select()
enters_list = list(enters)
print(enters_list)

db.close()
