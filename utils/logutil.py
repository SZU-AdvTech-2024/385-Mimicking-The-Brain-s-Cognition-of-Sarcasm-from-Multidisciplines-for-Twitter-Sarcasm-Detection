import logging
from logging.handlers import RotatingFileHandler
import os

class LoggerConfig:
    _instance = None
    _handler = None

    def __new__(cls, log_dir, log_file_name='train.log', max_bytes=200*1024*1024, backup_count=5):
        if cls._instance is None:
            cls._instance = super(LoggerConfig, cls).__new__(cls)
            # 确保日志目录存在
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # 定义日志文件路径
            log_file_path = os.path.join(log_dir, log_file_name)

            # 创建一个handler，用于写入同一个日志文件
            cls._handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=backup_count)
            cls._handler.setLevel(logging.INFO)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # cls._handler.setFormatter(formatter)
        return cls._instance

    def get_handler(self):
        return self._handler



def create_logger(name, logger_config):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logger_config.get_handler())
    return logger

if __name__ == '__main__':
    logger_config = LoggerConfig('./')
    logger = create_logger(__file__, logger_config)
    #
    logger.info('This is an info message from logger1.')
    # logger2.info('This is an info message from logger2.')