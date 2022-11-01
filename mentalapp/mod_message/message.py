"""File defining the creation class of each data on dataset"""


class Message:
    """Class containing one case of dataset"""

    def __init__(self, dict_data):
        """
        Init function of Message class
        :param dict_data: One line of raw data with text and classification (class)
        :type: dict
        :return: Nothing
        """
        self.text = dict_data['text']
        self.type_class = dict_data['class']
