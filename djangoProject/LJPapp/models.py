from django.db import models


class User(models.Model):
    iduser = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=45)
    password = models.CharField(max_length=45)

    class Meta:
        managed = True
        db_table = 'user'
