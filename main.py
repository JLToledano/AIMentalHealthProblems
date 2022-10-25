"""Main application file. Contains the general operating logic."""

from menu import option_menu, welcome_menu


def main():
    """Main function of the application. Manages the menu and calls to the main blocks.
    :return: Nothing
    """
    welcome_menu()
    option_menu()

main()