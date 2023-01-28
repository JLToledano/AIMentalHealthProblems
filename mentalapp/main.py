"""Main application file. Contains the general operating logic."""
import __init__ as init
import sys
from menu import option_menu, welcome_menu
from mod_dataset.dataset import Dataset


configuration_main = init.load_config_mentalapp()


def dataset_initialize():
    """
    Initialisation of the default dataset
    :return: Complete dataset with correct format
    :rtype: Dataset
    """
    complete_dataset = Dataset()
    raw_data = complete_dataset.read_file(configuration_main['FILE_DATASET_NAME'])
    complete_dataset.format_dataset(raw_data)

    return complete_dataset


def main():
    """Main function of the application. Manages the menu and calls to the main blocks
    :return: Nothing
    """
    welcome_menu()
    dataset = dataset_initialize()
    option_menu()

if __name__ == "__main__":
    main()
