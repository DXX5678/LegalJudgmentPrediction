from django.urls import path
from . import views

urlpatterns = [path('', views.login, name='login'), path('LJP/', views.index, name='index'),
               path('register/', views.register, name='register'), path('LJP/pre', views.predicate, name='predicate')]
