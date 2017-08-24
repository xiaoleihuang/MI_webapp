from django.conf.urls import url

from . import views

urlpatterns = [
	url(r'^test$', views.test, name='test'),
	url(r'^attention_form_calc/$', views.attention_form_calc, name='attention_form_calc'),
	url(r'^$', views.index, name='home'),
]
