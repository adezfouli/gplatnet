import logging

from latnet.util import get_git


class Logger:
    """
    This is a logger class used for logging the output of the algorithm.
    """


    @staticmethod
    def init_logger(path=None, name=None, logger_name=None):
        if logger_name is None:
            logger_name = __name__
        Logger.logger = logging.getLogger(logger_name)
        Logger.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if path is not None:
            fh = logging.FileHandler(path + '/' + name + '.log')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            Logger.logger.addHandler(fh)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        Logger.logger.addHandler(ch)

        git_hash, git_branch = get_git()
        Logger.logger.debug("logging started for git hash:" + str(git_hash) + " branch:" + str(git_branch))

    @staticmethod
    def remove_handlers():
        Logger.logger.handlers = []


    @staticmethod
    def get_string_logger():
        log_capture_string = StringBuffer()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        Logger.logger.addHandler(ch)
        return log_capture_string
