import os
import shutil

from django.db import models

from .mixins import MixinRegularCustomerInfo


class Species(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'<Species {self.name!r}>'


class BaseCustomer(models.Model):
    class Meta:
        abstract = True

    SEX_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female')
    ]

    name = models.CharField(max_length=100, null=False, default='<Unknown>')
    sex = models.CharField(max_length=1, null=False, choices=SEX_CHOICES)
    first_visit = models.DateTimeField(verbose_name='Date of first visit', auto_now_add=True)
    last_visit = models.DateTimeField(verbose_name='Date of last visit', auto_now=True)
    spent = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)


class Customer(BaseCustomer):
    class Meta(BaseCustomer.Meta):
        db_table = 'customer'

    favorite_species = models.ManyToManyField(Species)

    @classmethod
    def create(cls, **kwargs):
        new_customer = cls(**kwargs)
        new_customer.save()
        try:
            os.mkdir(os.path.join('face_detection', 'dataset', str(new_customer.pk)))
        except OSError as e:
            print(e)
        return new_customer

    def delete(self, using=None, keep_parents=False):
        dir = os.path.join('face_detection', 'dataset', str(self.pk))
        if os.path.exists(dir):
            shutil.rmtree(dir)

        # Ensure to return whatever overriden delete should return
        return super().delete(using=using, keep_parents=keep_parents)

    # <Magic Methods>
    def __str__(self):
        return f'{self.name}, ID {self.pk}'

    def __repr__(self):
        return f'<Customer {self.pk} - {self.name}>'


class RegularCustomer(BaseCustomer, MixinRegularCustomerInfo):
    class Meta(BaseCustomer.Meta):
        db_table = 'regular_customer'

    def __str__(self):
        if self.sex == 'M':
            return f'Mr. {self.name}, ID {self.pk}'
        else:
            return f'Mrs. {self.name}, ID {self.pk}'

    def __repr__(self):
        return f'<RegularCustomer {self.pk} - {self.name}>'
