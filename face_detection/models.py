from django.db import models

from .mixins import MixinRegularCustomerInfo


class BaseCustomer(models.Model):
    class Meta:
        abstract = True

    SEX_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female')
    ]

    name = models.CharField(max_length=100, null=False, default='<Unknown>')
    sex = models.CharField(max_length=1, null=False, choices=SEX_CHOICES)
    first_visit = models.DateTimeField(verbose_name='Date of first visit', auto_now=True)
    spent = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)


class Customer(BaseCustomer):
    class Meta(BaseCustomer.Meta):
        db_table = 'customer'

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
