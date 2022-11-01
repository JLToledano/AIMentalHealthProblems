"""File defining the creation class of the dataset"""

import csv
import os

from mentalapp.mod_message.message import Message


class Dataset:
    """Class containing the complete dataset"""

    files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
    dataset = []

    def read_file(self, name_file):
        """
        Function to read Suicide Detection file which contains raw datas.
        :param name_file: Name of file read
        :rtype: String
        :return: lines of file read.
        :rtype: list[dict[String:String]
        """
        read_lines = []
        file_path = os.path.join(self.files_path, f'{name_file}')

        with open(file_path, encoding='utf-8') as file:
            content = csv.DictReader(file)
            for line in content:
                read_lines.append(line)

        return read_lines

    def format_dataset(self, list_data):
        """
        Function to transform raw data into dataset object
        :param list_data: List with raw data of dataset
        :rtype: list[dict[String:String]]
        :return: Nothing
        """
        self.dataset = list(map(lambda line: Message(line), list_data))
