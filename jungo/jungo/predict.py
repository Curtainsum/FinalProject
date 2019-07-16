from _sha1 import sha1
import cifar
from . import cache

CACHED = True


def get_hash(data):
    sha1_obj = sha1()
    sha1_obj.update(data)
    hash = sha1_obj.hexdigest()
    return hash


def do_predict(img_data):
    is_success, type, sub_type = cifar.predict(img_data)
    return is_success, type, sub_type


def get_result(img_data):
    hash = get_hash(img_data)
    if not CACHED:
        return do_predict(img_data)
    else:
        is_exist, type, sub_type = cache.check_exists(hash)
        if not is_exist:
            is_success, type, sub_type = do_predict(img_data)
            if is_success:
                cache.add_result(hash, type, sub_type)
            return is_success, type, sub_type
        else:
            return  True,type,sub_type