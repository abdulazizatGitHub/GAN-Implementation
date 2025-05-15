import logging

from . import config


class Logger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)

        sh = logging.StreamHandler()
        sh.setFormatter(config.log_config.stream_formatter)
        sh.setLevel(config.log_config.level)
        self.addHandler(sh)

        fh = logging.FileHandler()
        fh.setFormatter(config.log_config.file_formatter)
        sh.setLevel(config.log_config.level)
        self.addHandler(fh)

    def turn_on(self):
        self.setLevel(config.log_config.level)
        for handler in self.handlers:
            handler.setLevel(config.log_config.level)
    
    def turn_off(self):
        self.setLevel(logging.CRITICAL + 1)
        for handler in self.handlers:
            handler.setLevel(logging.CRITICAL + 1)
