from django.shortcuts import render
from django.http import HttpResponse

# py4j library
from py4j.java_gateway import JavaGateway


# initialize the gateway
gateway = JavaGateway()

def index(request):
	
	return HttpResponse("Hello, world. You're at the polls index.")
