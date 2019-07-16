import pymysql

db_host = '39.105.214.35'
db_port = 3366
db_name = 'b10'
db_charset = 'utf8mb4'

db_user = 'b10'
db_pw = '1234'


def check_exists(hash):
    conn = pymysql.connect(user=db_user, password=db_pw,
                           host=db_host, port=db_port, db=db_name,
                           charset=db_charset)
    sql = 'select type,sub_type from cache where hash=\'{}\''.format(hash)
    cursor = conn.cursor()
    result = cursor.execute(sql)
    cache_item = cursor.fetchmany(result)
    conn.close()
    if len(cache_item) > 0:
        return True, cache_item[0][0], cache_item[0][1]
    else:
        return False, None, None


def add_result(hash, type, sub_type):
    conn = pymysql.connect(user=db_user, password=db_pw,
                           host=db_host, port=db_port, db=db_name,
                           charset=db_charset)
    sql = 'insert into cache (hash,type,sub_type) values(\'{}\',\'{}\',\'{}\')'.format(hash, type, sub_type)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    conn.close()
