from django.shortcuts import render
from django.http import HttpResponse
from userinput import UserInputForm 

def index(request):
    print("hey")
    if request.method == 'POST':
        form_class = UserInputForm(request.POST)

        if form_class.is_valid():
            print(form_class.cleaned_data['text'])
        else:
            print("data was invalid")

    else:
        form_class = UserInputForm
        
    return render(request, 'index.html', {
        'form': form_class
        })

def chat(request):
    if request.method == 'POST':
        form_class = UserInputForm(request.POST)

        if form_class.is_valid():
            print(form_class.cleaned_data['text'])
        else:
            print("invalid data")

    return render(request, 'index.html', {
        'form': form_class
        })