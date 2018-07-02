"""
Simply drop all tables created

Another way is use raw SQL:
db.execute_sql('drop table person')


"""

import peewee
from tables import Person, GoIn

# drop table
try:
    Person.drop_table()
    print("[db] table People drop")

    GoIn.drop_table()
    print("[db] table GoIn drop")


except peewee.OperationalError:
    print("[db] Error, cannot drop tables.")


