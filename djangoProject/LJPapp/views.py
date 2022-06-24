from django.contrib import messages
from django.shortcuts import render, redirect, reverse
from django.views.decorators.csrf import csrf_exempt
from LJPapp.models import User
from LJPapp.test import t_predicate


@csrf_exempt
def login(request):
    if request.method == 'POST':
        concat = request.POST
        username = concat['username']
        password = concat['password1']
        one = User.objects.all()
        num = one.count()
        user = User(num+1, username, password)
        user.save()
    return render(request, "index.html")


def register(request):
    return render(request, "register.html")


def index(request):
    message = {}

    concat = request.POST
    username = concat['u']
    password = concat['p']
    try:
        one = User.objects.get(name=username)
        if one.password == password:
            return render(request, "predicate.html")
        else:
            message["msg"] = "密码错误！"
            return render(request, "index.html", message)
    except Exception as r:
        print(r)
        message["msg"] = "用户名不存在！"
        return render(request, "index.html", message)


def predicate(request):
    context = {}
    if 'input' in request.GET and request.GET['input']:
        fact = request.GET['input']
        context["fact"] = fact
        try:
            result = t_predicate(fact)
            context["result"] = result
            messages.success(request, "模型分析中，请稍后。")
        except Exception as r:
            print(r)
            messages.error(request, "错误警告！")
    return render(request, "predicating.html", context)
# Create your views here.
