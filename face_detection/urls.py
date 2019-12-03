from django.urls import path

from . import views as _views


urlpatterns = [
    path('index/', _views.index, name='index'),
    path('customer/<int:pk>/', _views.CustomerDetailView.as_view(), name='customer_detail'),
]
