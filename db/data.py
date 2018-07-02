import peewee
from tables import Person, GoIn
from datetime import date


# insert people
p1 = Person(name="Chau")
p1.save()

p2 = Person(name="Chau2")
p2.save()

# insert goIn
g1 = GoIn(person=p1, n_enter=10)
g1.save()

g2 = GoIn(person=p2, n_enter=15)
g2.save()

print("[db] Done insert data.")

