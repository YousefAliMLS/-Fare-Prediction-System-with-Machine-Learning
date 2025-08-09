from django.contrib import admin
from django.urls import path
from predictor.views import predict_fare

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', predict_fare),
]
