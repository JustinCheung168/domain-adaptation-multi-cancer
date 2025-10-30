import datetime
import logging as py_logging
import os

from transformers import logging as hf_logging

_logging_initialized = False
def init_logging(logging_dir: str):
    """Set up global logging format with timestamps. Assume a logger already exists."""
    # Don't initialize more than once
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    # Create a timestamped filename
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(logging_dir, exist_ok=True)
    log_filename = os.path.join(logging_dir, f"log_{current_time}.log")

    file_handler = py_logging.FileHandler(log_filename)
    file_handler.setLevel(py_logging.DEBUG)
    file_handler.setFormatter(py_logging.Formatter(
        '[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s >>> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Attach to the root logger
    root_logger = py_logging.getLogger()
    root_logger.setLevel(py_logging.DEBUG)
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)

    # Attach to transformers logger explicitly
    transformers_logger = py_logging.getLogger("transformers")
    transformers_logger.setLevel(py_logging.DEBUG)
    transformers_logger.propagate = True
    if file_handler not in transformers_logger.handlers:
        transformers_logger.addHandler(file_handler)

    hf_logging.set_verbosity_debug()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    print(f"Logging to {log_filename}")




class LogMixin:
    """Make this one of the parents of a class to have logs created in the class appear."""
    def __init__(self):
        self.logger = py_logging.getLogger(self.__class__.__name__)
