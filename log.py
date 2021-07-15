import os
import logging
import time


class CreateLog():

    def __init__(self, LOG_DIR):
        self.time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.head = '%(asctime)-15s %(message)s'

        logging.basicConfig(filename=os.path.join(LOG_DIR, f"{self.time_str}.log"),
                            format=self.head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

    def write(self, string):
        logging.info(string)