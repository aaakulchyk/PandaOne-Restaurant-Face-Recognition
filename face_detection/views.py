from django.shortcuts import render
from django.views import generic

from . import models as _models


def index(request, *args, **kwargs):
    return render(request, 'face_detection/index.html')


class CustomerDetailView(generic.DetailView):
    model = _models.Customer
    template_name = 'face_detection/customer_detail.html'
    context_object_name = 'customer'
