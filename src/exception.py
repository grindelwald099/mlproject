
import sys
import logging

def error_message_detail(error, error_detail: sys):
    """Return detailed information about an error."""
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in file: {file_name} "
        f"line number: {line_no} "
        f"error message: {str(error)}"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message
        self.error_detail = error_detail

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0

    except Exception as e:
        logging.info("Divide by Zero")

        error_msg = error_message_detail(e, sys)
        raise CustomException(error_msg, sys) from e
