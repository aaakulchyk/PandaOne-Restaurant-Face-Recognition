from django.db import models


class MixinRegularCustomerInfo(models.Model):
    birthday = models.DateField(verbose_name='Date of birth', null=True)
    favorite_table = models.PositiveIntegerField(null=True)
    has_pet = models.BooleanField(default=False)
