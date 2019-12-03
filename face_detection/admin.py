from django.contrib import admin

from . import models as _models


admin.site.register(_models.Customer)
admin.site.register(_models.RegularCustomer)
