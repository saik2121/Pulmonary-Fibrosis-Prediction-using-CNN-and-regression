from django.urls import path
from .views import *

urlpatterns = [
    path('',index,name='home'),
    path('pulmonaryfibrosis/',pulmonaryfibrosis,name='pfp'),
    path('progression/',getprogression,name='gp'),
]
