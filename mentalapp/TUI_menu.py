"""File containing the console message printing functions"""

from rich.console import Console
from rich.markdown import Markdown
from rich.align import Align
from rich.panel import Panel

def welcome_menu():
    """
    Function which prints the welcome message when starting the application
    :return: Nothing
    """

    console = Console()

    AI_message = """
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m      - - -         - - - - - - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m    -       -       -               -m- - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m  -     -     -     - - - -   - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-     -   -     -         -   -      m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-     - - -     -         -   -      m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-               -         -   -      m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-   - - - - -   -         -   -      m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-   -       -   -   - - - -   - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-   -       -   -   -               -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m- - -       - - -   - - - - - - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """

    mental_message = """
    - - - - - - - - m-               -   - - - - - - - - -   - -         - - -   - - - - - - - - -         - - -         - - -            m- - - - - - - -
    - - - - - - - - m- -           - -   -               -   -   -       -   -   -               -       -       -       -   -            m- - - - - - - -
    - - - - - - - - m-   -       -   -   -   - - - - - - -   -     -     -   -   -               -     -     -     -     -   -            m- - - - - - - -
    - - - - - - - - m-     -   -     -   -   -               -       -   -   -   - - - -   - - - -   -     -   -     -   -   -            m- - - - - - - -
    - - - - - - - - m- -     -     - -   -   - - - - - - -   -   -     - -   -         -   -         -     - - -     -   -   -            m- - - - - - - -
    - - - - - - - - m- - -       - - -   -   - - - - - - -   -   - -     -   -         -   -         -               -   -   -            m- - - - - - - -
    - - - - - - - - m- -   -   -   - -   -   -               -   -   -       -         -   -         -   - - - - -   -   -   -            m- - - - - - - -
    - - - - - - - - m- -     -     - -   -   - - - - - - -   -   -     -     -         -   -         -   -       -   -   -   - - - - - - -m- - - - - - - -
    - - - - - - - - m- -           - -   -               -   -   -       -   -         -   -         -   -       -   -   -               -m- - - - - - - -
    - - - - - - - - m- -           - -   - - - - - - - - -   - - -         - -         - - -         - - -       - - -   - - - - - - - - -m- - - - - - - -
    """

    health_message = """
    - - - - - - - - m- - - -   - - - -   - - - - - - - - -         - - -         - - -               - - - - - - - - -   - - - -   - - - -m- - - - - - - -
    - - - - - - - - m-     -   -     -   -               -       -       -       -   -               -               -   -     -   -     -m- - - - - - - -
    - - - - - - - - m-     -   -     -   -   - - - - - - -     -     -     -     -   -               -               -   -     -   -     -m- - - - - - - -
    - - - - - - - - m-     - - -     -   -   -               -     -   -     -   -   -               - - - -   - - - -   -     - - -     -m- - - - - - - -
    - - - - - - - - m-               -   -   - - - - - - -   -     - - -     -   -   -                     -   -         -               -m- - - - - - - -
    - - - - - - - - m-               -   -   - - - - - - -   -               -   -   -                     -   -         -               -m- - - - - - - -
    - - - - - - - - m-     - - -     -   -   -               -   - - - - -   -   -   -                     -   -         -     - - -     -m- - - - - - - -
    - - - - - - - - m-     -   -     -   -   - - - - - - -   -   -       -   -   -   - - - - - - -         -   -         -     -   -     -m- - - - - - - -
    - - - - - - - - m-     -   -     -   -               -   -   -       -   -   -               -         -   -         -     -   -     -m- - - - - - - -
    - - - - - - - - m- - - -   - - - -   - - - - - - - - -   - - -       - - -   - - - - - - - - -         - - -         - - - -   - - - -m- - - - - - - -
    """


    AI_message_align = Align.center(AI_message, vertical="middle")
    mental_message_align = Align.center(mental_message, vertical="middle")
    health_message_align = Align.center(health_message, vertical="middle")


    console.print(AI_message_align, style="bold blue")
    console.print(mental_message_align, style="bold blue")
    console.print(health_message_align, style="bold blue")


    welcome_message= """# Bienvenido a la aplicación"""
    welcome_message_markdown = Markdown(welcome_message)
    console.print(welcome_message_markdown)


def option_menu():
    """
    Function which prints the options message
    :return: Nothing
    """

    console = Console()

    option_message = """Escoge una de las siguientes opciones"""
    option_message_align = Align(option_message, align="left")
    console.print(Panel.fit(option_message_align, style="bold"))

    first_option_message = """1. Utilizar modelo preentrenado"""
    first_option_message_markdown = Markdown(first_option_message)
    console.print(first_option_message_markdown, style="bold")

    second_option_message = """2. Entrenar modelo"""
    second_option_message_markdown = Markdown(second_option_message)
    console.print(second_option_message_markdown, style="bold")

    third_option_message = """3. Evaluar modelo"""
    third_option_message_markdown = Markdown(third_option_message)
    console.print(third_option_message_markdown, style="bold")

    fourth_option_message = """4. Personalizar parámetros entrenamiento"""
    fourth_option_message_markdown = Markdown(fourth_option_message)
    console.print(fourth_option_message_markdown, style="bold")

    fifth_option_message = """5. Personalizar ruta dataset"""
    fifth_option_message_markdown = Markdown(fifth_option_message)
    console.print(fifth_option_message_markdown, style="bold")

    sixth_option_message = """6. Ayuda"""
    sixth_option_message_markdown = Markdown(sixth_option_message)
    console.print(sixth_option_message_markdown, style="bold")

    seventh_option_message = """7. Salir"""
    seventh_option_message_markdown = Markdown(seventh_option_message)
    console.print(seventh_option_message_markdown, style="bold")

    console.print("")


