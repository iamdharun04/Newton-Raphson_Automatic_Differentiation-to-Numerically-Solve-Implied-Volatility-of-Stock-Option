from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('calculate/', views.calculate_option, name='calculate_option'),
    path('project-details/', views.project_details, name='project_details'),
]
