"""File containing the console message printing functions"""
import os
import msvcrt

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.align import Align
from rich.panel import Panel
from rich.theme import Theme
from rich.table import Table

def clear_console():
    """
    Function to clean the execution console
    :return: Nothing
    """

    #If the system on which it runs is windows
    if os.name == "nt":
        os.system("cls")
    #If the system on which it is running is linux
    else:
        os.system("clear")


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


def help_menu():
    """
    Function which prints help messages
    :return: Nothing
    """
    
    #Customization of the console with special predefined styles
    custom_theme = Theme({"success": "green", "error": "red", "option":"yellow", "required_parameter":"purple"})
    console = Console(theme = custom_theme)

    #Design and printing of welcome message
    welcome_message= """# Bienvenido a la aplicación. Gracias a esta herramienta podrás utilizar modelos de Inteligencia Artificial para detectar señales de suicidio en textos"""
    welcome_message_markdown = Markdown(welcome_message)
    console.print(welcome_message_markdown)

    #Design and printing of the liability warning message
    console.print(Panel.fit("[error]WARNING[/error]", border_style="red"))
    console.print("""[error]Los resultados que pueda reflejar esta aplicación solamente deben ser tomados en consideración por especialistas junto con pruebas externas que reflejen las intenciones de suicidio o no del escritor del mensaje procesado.[/error]""")
    console.print("""[error]Esta es una aplicación con intenciones educativas. En caso de detección de problemas en la salud mental de un usuario por favor acudir a un especialista.[/error]""")
    console.print(Panel.fit("[error]WARNING[/error]", border_style="red"))
    console.print()

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 1
    first_option_table = Table(expand = True)
    first_option_table.add_column("[option]OPCIÓN 1[/option] Utilizar modelo preentrenado", justify="full")
    first_option_table.add_row(
    """En esta opción se podrá elegir una de las redes neuronales pre-entrenadas disponibles en la aplicación. Después de efectuar la elección, se pedirá un texto que deberá escribirse en inglés y el cuál representa un mensaje que será analizado por la red neuronal. El resultado del análisis reflejará si el mensaje tiene señales de una posible intención de suicidio o no."""
    )
    console.print(first_option_table)

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 2
    second_option_table = Table(expand = True)
    second_option_table.add_column("[option]OPCIÓN 2[/option] Entrenar modelo", justify = "full")
    second_option_table.add_row(
    """En esta opción se podrá elegir una base pre-entrenada como modelo de red neronal a la que se le añaden capas extras para su adaptación al uso dado en esta aplicación. Después de la elección se entrenará el modelo con el dataset proporcionado y se almacenará el modelo entrenado en un fichero con el nombre especificado por el usuario para su posible uso posterior.\n"""
    + """\n""" +
    """Por defecto el dataset proporcionado contiene datos que se destinarán a entrenamiento y datos que se destinarán a evaluación."""
    )
    console.print(second_option_table)

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 3
    third_option_table = Table(expand = True)
    third_option_table.add_column("[option]OPCIÓN 3[/option] Evaluar modelo", justify = "full")
    third_option_table.add_row(
    """En esta opción se podrá elegir un modelo pre-entrenado disponible en la aplicación para evaluar su rendimiento con el dataset de datos proporcionado.\n"""
    + """\n""" +
    """Por defecto el dataset proporcionado contiene datos que se destinarán a entrenamiento y datos que se destinarán a evaluación."""
    )
    console.print(third_option_table)

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 4
    fourth_option_table_parameters = Table(box = box.HEAVY)
    fourth_option_table_parameters.add_column("Parameter", justify = "center")
    fourth_option_table_parameters.add_column("Default value", justify = "center")
    fourth_option_table_parameters.add_column("Definition", justify = "full")
    fourth_option_table_parameters.add_column("Considerations", justify = "center")
    
    fourth_option_table_parameters.add_row("MAX_DATA_LEN", "200", "Número de palabras máximo que admite el modelo neuronal. El resto de palabras se truncan", "")
    fourth_option_table_parameters.add_row("BATCH_SIZE", "16", "Tamaño de lote. Número de datos que se insertan en la red neuronal cada vez", "")
    fourth_option_table_parameters.add_row("EPOCHS", "3", "Número de épocas (iteraciones)", "")
    fourth_option_table_parameters.add_row("DATALOADER_NUM_WORKERS", "1", "Número de procesos que se ejecutan en paralelo. Se analizan X datos en paralelo", "")
    fourth_option_table_parameters.add_row("DROP_OUT_BERT", "0.3", "Tanto por uno de neuronas que se desactivan aleatoriamente en una de las capas de la red nueronal", "[option]APLICABLE EN BERT[/option]")

    fourth_option_table = Table(expand = True)
    fourth_option_table.add_column("[option]OPCIÓN 4[/option] Personalizar parámetros entrenamiento", justify = "full")
    fourth_option_table.add_row(
    """En esta opción el usuario podrá cambiar la configuración inicial de los parámetros de división del dataset de datos, entrenamiento y evaluación.\n"""
    + """\n""" +
    """Los cambios realizados perduran mientras se esté ejecutando la aplicación. El reinicio de esta significa el reseteo de los cambios realizados.\n"""
    + """\n""" +
    """La configuración inicial es la siguiente:\n"""
    )
    fourth_option_table.add_row(fourth_option_table_parameters)
    console.print(fourth_option_table)
    
    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 5
    fifth_option_table = Table(expand = True)
    fifth_option_table.add_column("[option]OPCIÓN 5[/option] Personalizar ruta dataset", justify = "full")
    fifth_option_table.add_row(
    """En esta opción el usuario puede cambiar la ruta predefinida del dataset para utilizar uno configurado por el usuario y que esté situado en otro directorio del ordenador.\n"""
    + """\n""" +
    """La extensión del archivo deberá ser .csv y con el siguiente formato de mensajes:\n"""
    + """\n""" +
    """             [required_parameter]Número del mensaje[/required_parameter],[required_parameter]"Texto a analizar entre comillas dobles"[/required_parameter],[required_parameter]clasificación[/required_parameter]\n"""
    + """\n""" +
    """La clasificación deberá ser: [option]suicide[/option] o [option]non-suicide[/option]. El texto a analizar puede tener varios párrafos separados por retorno de carro."""
    )
    console.print(fifth_option_table)

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 6
    sixth_option_table = Table(expand = True)
    sixth_option_table.add_column("[option]OPCIÓN 6[/option] Ayuda", justify = "full")
    sixth_option_table.add_row(
    """Con esta opción se imprime por pantalla la guía de ayuda al usuario de la aplicación. La guía que se muestra en este momento es la que está disponible.\n"""
    + """\n""" +
    """Para más información, acudid a la documentación externa asociada al aplicativo."""
    )
    console.print(sixth_option_table)

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #Design and printing of the explanatory message menu option 7
    seventh_option_table = Table(expand = True)
    seventh_option_table.add_column("[option]OPCIÓN 7[/option] Salir", justify = "full")
    seventh_option_table.add_row(
    """Con está opción finaliza la ejecución de la aplicación.\n"""
    + """\n""" +
    """Los modelos entrenados persistirán al apagado de la aplicación pero no los cambios realizados en la configuración de entrenamiento."""
    )
    console.print(seventh_option_table)

    #It waits for user to press a key to continue displaying menu
    msvcrt.getch()

    #The console is cleared after entire help menu has been displayed
    clear_console()
