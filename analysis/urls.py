from django.urls import path
from .views import PricePredictionView,DataSummaryView,PriceDistributionView,ModelEvaluationView,PlotActualVsPredictedView,PlotPriceByHeightView

urlpatterns = [
    path('predict-price/', PricePredictionView.as_view(), name='predict-price'),
    path('data-summary/', DataSummaryView.as_view(), name='data-summary'),
    path('price-distribution/', PriceDistributionView.as_view(), name='price-distribution'),
    path('model-evaluation/', ModelEvaluationView.as_view(), name='model-evaluation'),
    path('plot-price-by-height/', PlotPriceByHeightView.as_view(), name='plot-price-by-height'),
    path('plot-actual-vs-predicted/', PlotActualVsPredictedView.as_view(), name='plot-actual-vs-predicted'),
]