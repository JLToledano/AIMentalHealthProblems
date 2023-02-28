"""File to select different models and prepare the appropriate configuration"""

from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.theme import Theme

from mod_BERT.model_BERT import BERTSentimentClassifier
from mod_distilBERT.model_distilBERT import DistilBERTSentimentClassifier
from mod_alBERT.model_alBERT import AlBERTSentimentClassifier

def model_selector(configuration):
    """
    User chooses neural network model and its configuration is set up
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: Basic Model configured (if applicable)
    :type: MODELSentimentClassifier
    :return: Function that transforms input data into special codes (tokens)
    :type: Tokenizer
    """

    custom_theme = Theme({"success":"green", "error":"red", "option":"yellow"})
    console = Console(theme = custom_theme)
    
    model_selection = False
    pre_trained_model_configurations = {
        'BERT': 'BERT_configurations(configuration)',
        #'DISTILBERT': 'DISTILBERT_configurations(configuration)'
        'ALBERT': 'AlBERT_configurations(configuration)'
    }
    
    #As long as a pre-trained model has not been selected
    while not model_selection:
        console.print(Panel.fit("TECNOLOGÍAS DISPONIBLES"))
        #Available options are printed
        for pre_trained_model in pre_trained_model_configurations.keys():
            console.print("[option]" + pre_trained_model + "[/option]")

        console.print("")
        selected_option = Prompt.ask("Seleccione una opción")

        #User option is validated
        if selected_option in pre_trained_model_configurations.keys():
            model_selection = True
        else:
            console.print("[error]Opción incorrecta[/error], por favor seleccione una de las [success]opciones disponibles[/success].\n")

    #Configured model and tokenizer are returned.
    model_configuration = eval(pre_trained_model_configurations[selected_option])

    return model_configuration
    


def BERT_configurations(configuration):
    """
    Configuration of the BERT model and tokeniser
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: BERT model and BERT tokenizer
    :type: dict[String:MODELSentimentClassifier/Tokenizer]
    """

    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = BertTokenizer.from_pretrained(configuration['PRE_TRAINED_MODEL_NAME']['BERT'])
    
    #Creation of BERT model
    model = BERTSentimentClassifier(configuration['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration['PRE_TRAINED_MODEL_NAME']['BERT'], configuration['DROP_OUT_BERT'])

    BERT_configuration = {}
    BERT_configuration['model'] = model
    BERT_configuration['tokenizer'] = tokenizer
    BERT_configuration['name_model'] = 'BERT'

    return BERT_configuration


def DISTILBERT_configurations(configuration):
    """
    Configuration of the DistilBERT model and tokeniser
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: DistilBERT model and DistilBERT tokenizer
    :type: dict[String:MODELSentimentClassifier/Tokenizer]
    """

    #Function transforming input data into special codes (tokens) for DistilBERT model
    tokenizer = DistilBertTokenizer.from_pretrained(configuration['PRE_TRAINED_MODEL_NAME']['DistilBERT'])
    
    #Creation of DistilBERT model
    model = DistilBERTSentimentClassifier(configuration['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration['PRE_TRAINED_MODEL_NAME']['DistilBERT'], configuration['DROP_OUT_BERT'])

    DistilBERT_configuration = {}
    DistilBERT_configuration['model'] = model
    DistilBERT_configuration['tokenizer'] = tokenizer
    DistilBERT_configuration['name_model'] = 'DistilBERT'

    return DistilBERT_configuration


def AlBERT_configurations(configuration):
    """
    Configuration of the AlBERT model and tokeniser
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: AlBERT model and AlBERT tokenizer
    :type: dict[String:MODELSentimentClassifier/Tokenizer]
    """

    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = AlbertTokenizer.from_pretrained(configuration['PRE_TRAINED_MODEL_NAME']['AlBERT'])
    
    #Creation of BERT model
    model = AlBERTSentimentClassifier(configuration['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration['PRE_TRAINED_MODEL_NAME']['AlBERT'], configuration['DROP_OUT_BERT'])

    AlBERT_configuration = {}
    AlBERT_configuration['model'] = model
    AlBERT_configuration['tokenizer'] = tokenizer
    AlBERT_configuration['name_model'] = 'AlBERT'

    return AlBERT_configuration