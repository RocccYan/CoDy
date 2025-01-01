
import logging
import json


def load_json(filepath):
    with open(filepath,'r', encoding='utf-8') as f:
        return json.load(f)

def save_as_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_json)
    print(f"{filename} saved.")

def default_json(t):
    return f'{t}'

def setup_logger(logger_name="", log_file="demo.log", level=logging.INFO):
    """
    Sets up a logger with a console handler and a file handler.

    :param logger_name: The name of the logger.
    :param log_file: The filename of the log file.
    :param level: The logging level (e.g., logging.INFO).
    :return: The configured logger.
    """
    # Step 1: Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)  # Set the logging level

    # Step 2: Create a console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Step 3: Create a file handler and set level to DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Store all messages in the file

    # Step 4: Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Step 5: Add the formatter to the handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Step 6: Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # Reset counter if validation loss improves
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping triggered')
                self.early_stop = True


