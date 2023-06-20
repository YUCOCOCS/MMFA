'''
Function:
    define the logging function
Author:
    Zhenchao Jin
'''
import logging


'''log function.'''
class LLogger():
    def __init__(self, logfilepath, **kwargs):
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.FileHandler(logfilepath), logging.StreamHandler()] #写到对应的文件中
        ) # 使用默认格式化程序创建 StreamHandler 并将其添加到根日志记录器中，从而完成日志系统的基本配置
    @staticmethod
    def log(level, message):
        logging.log(level, message)
    @staticmethod
    def debug(message):
        LLogger.log(logging.DEBUG, message)
    @staticmethod
    def info(message):
        LLogger.log(logging.INFO, message)
    @staticmethod
    def warning(message):
        LLogger.log(logging.WARNING, message)
    @staticmethod
    def error(message):
        LLogger.log(logging.ERROR, message)