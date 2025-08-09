import pickle
import numpy as np
import os
from django.conf import settings
from django.shortcuts import render

model_path = os.path.join(settings.BASE_DIR, 'fare_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def predict_fare(request):
    if request.method == 'POST':
        try:
            features = [
                float(request.POST.get('pickup_longitude')),
                float(request.POST.get('pickup_latitude')),
                float(request.POST.get('dropoff_longitude')),
                float(request.POST.get('dropoff_latitude')),
                int(request.POST.get('passenger_count')),
                int(request.POST.get('hour')),
                int(request.POST.get('day')),
                int(request.POST.get('month')),
                int(request.POST.get('weekday')),
                int(request.POST.get('year')),
                float(request.POST.get('jfk_dist')),
                float(request.POST.get('ewr_dist')),
                float(request.POST.get('lga_dist')),
                float(request.POST.get('sol_dist')),
                float(request.POST.get('nyc_dist')),
                float(request.POST.get('distance')),
                float(request.POST.get('bearing')),
                int(request.POST.get('is_weekend')),
                int(request.POST.get('rush_hours')),
                int(request.POST.get('car_condition_Excellent', 0)),
                int(request.POST.get('car_condition_Good', 0)),
                int(request.POST.get('car_condition_Very Good', 0)),
                int(request.POST.get('weather_rainy', 0)),
                int(request.POST.get('weather_stormy', 0)),
                int(request.POST.get('weather_sunny', 0)),
                int(request.POST.get('weather_windy', 0)),
                int(request.POST.get('traffic_condition_Dense Traffic', 0)),
                int(request.POST.get('traffic_condition_Flow Traffic', 0))
            ]

            prediction = model.predict([features])[0]
            return render(request, 'predictor/form.html', {'result': round(prediction, 2)})

        except Exception as e:
            return render(request, 'predictor/form.html', {'error': str(e)})

    return render(request, 'predictor/form.html')
