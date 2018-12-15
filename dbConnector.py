#encoding:utf-8
import sys
from imp import reload
reload(sys)
import pymysql

class DBConnector:
    def dbConnector(self):
        self.connect = pymysql.Connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='123456',
        db='xinwentest',
        charset='utf8'
        )
        return self.connect
