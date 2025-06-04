import logging
import os
from datetime import datetime

Log_file = f'{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log'
Log_path = os.path.join(os.getcwd(), "Logs")

os.makedirs(Log_path, exist_ok=True)

Log_file_path = os.path.join(Log_path, Log_file)

logging.basicConfig(
    filename=Log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
