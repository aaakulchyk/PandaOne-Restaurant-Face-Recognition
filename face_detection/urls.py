from django.urls import path

from . import views as _views


urlpatterns = [
    path('index/', _views.index, name='index'),
    path('', _views.face_recognition, name='face_recognition'),
    path('detect/', _views.detect_async, name='detect'),
    path('json/customer/<int:pk>/', _views.json_customer, name='json_customer'),
    path('customer/', _views.CustomerListView.as_view(), name='customer_list'),
    path('customer/<int:pk>/', _views.CustomerDetailView.as_view(), name='customer_detail'),
    path('regular_customer/', _views.RegularCustomerListView.as_view(), name='regular_customer_list'),
    path('regular_customer/<int:pk>/', _views.RegularCustomerDetailView.as_view(), name='regular_customer_detail'),
    path('test/', _views.test, name='test'),
    path('contacts/', _views.contacts, name='contacts'),
]
