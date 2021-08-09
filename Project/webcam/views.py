from django.http import HttpResponse, StreamingHttpResponse
from django.template import loader
import time
from django.views.decorators import gzip

# xac dinh cac url chinh
def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render({}, request))

def contact(request):
    template = loader.get_template('contact.html')
    return HttpResponse(template.render({}, request))

def camera(request):
    template = loader.get_template('camera.html')
    return HttpResponse(template.render({}, request))


def stream():
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')
        time.sleep(1)


@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
    