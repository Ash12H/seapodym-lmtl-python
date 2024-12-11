import logging
import warnings

# TODO(Jules): The logger is not working as expected. Especially when multiprocessing is on.
# # For now debug is commented out.


def indent_message(msg: str) -> list[str]:
    return "".join(["\n|\t" + line for line in msg.split("\n")])


class CustomFormatter(logging.Formatter):
    """Colored and indented log messages."""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record) -> str:
        record.msg = indent_message(record.msg)
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("Seapodym")  # Root logger
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter(fmt="%(asctime)s :: %(name)s ::  %(levelname)s ::%(message)s\n"))
logger.addHandler(console_handler)


def set_critical() -> None:
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger("Seapodym").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")


def set_error() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("Seapodym").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def set_warning() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("Seapodym").setLevel(logging.WARNING)


def set_verbose() -> None:
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("Seapodym").setLevel(logging.INFO)


def set_debug() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("Seapodym").setLevel(logging.DEBUG)
