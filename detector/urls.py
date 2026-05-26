from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('about/', about, name='about'),
    path('model/', model, name='model'),
    path('login/', login_page, name='login'),
    path('dashboard/', dashboard, name='dashboard'),
    path('report/<int:id>/', report, name='report'),
    path('logout/', logout_page, name='logout'),
]