import os
import sys


def error_message_detail(error: Exception, error_detail) -> str:
    """
    Constructs a detailed error message using traceback info.
    """
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script [{file_name}] at line [{line_number}]: {str(error)}"
    else:
        return f"Error: {str(error)} (No traceback info available)"


class USvisaException(Exception):
    def __init__(self, error_message: Exception, error_detail):
        """
        Custom exception class that enhances the error message.
        """
        super().__init__(str(error_message))  # Call base Exception with string
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


