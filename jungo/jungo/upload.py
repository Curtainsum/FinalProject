from django.http import HttpResponse
from . import predict
import json


def upload_file(request):
    if request.method == "POST":
        myFile = request.FILES.get("file", None)
        data = myFile.read()
        is_success, type, sub_type = predict.get_result(data)

        if is_success:
            result = json.dumps(
                {
                    'code': 0,
                    'data':
                        {
                            'type': type,
                            'sub_type': sub_type
                        }
                })
            return HttpResponse(result, content_type="application/json")

    return HttpResponse(json.dumps({'code': 1}), content_type="application/json")
