from tables import Person

# perform query
print("[DEBUG] select * from Person")
query = Person.select()
for p in query:
    print(p.name)

