from django.urls import path
from .views import *


urlpatterns = [
    path('login', LoginView.as_view(), name='login'),
    path('logout', LogoutView.as_view(), name='logout'),
    path('doctor/register', RegisterDoctorView.as_view(), name='doctor-register'),
    path('doctor/profile/update/', EditDoctorProfileView.as_view(), name='doctor-profile-update'),    
]