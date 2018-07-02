"""
Testing peewee models
ORM: Object Relational Model
"""

import peewee
import datetime

# create
database = peewee.SqliteDatabase("wee.db")


###########################################################
class Person(peewee.Model):
    """
    ORM model of People table
    """

    # just name, no need id because peewee auto add
    # auto-incrementing integer primary key id
    name = peewee.CharField(unique=True)

    def __str__(self):
        return self.name

    class Meta:
        database = database


###########################################################
class GoIn(peewee.Model):
    """
    ORM Model of GoIn table
    """

    # foreign key
    person = peewee.ForeignKeyField(
        Person,
        on_delete='CASCADE'
    )
    # number of times entering
    n_enter = peewee.IntegerField(default=0)

    # date entering
    date_enter = peewee.DateField(default=datetime.date.today())

    def __str__(self):
        return "{}:{}".format(self.person.name, self.date_enter)

    class Meta:
        database = database


def main():
    try:
        Person.create_table()
        print("[db] Table Person created")

        GoIn.create_table()
        print("[db] Table GoIn created")

    except peewee.OperationalError:
        print("[db] Error cannot create tables.")


if __name__ == '__main__':
    main()
