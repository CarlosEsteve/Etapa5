from tabulate import tabulate

def tabl(df):
    print(tabulate(df.head(),headers='keys'))
    print(df.shape)

# Formato de los prints
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def headr(text):
    return ('\n'+color.UNDERLINE + text + color.END+'\n')


def titl(text):
    return ('\n'+color.BOLD+color.UNDERLINE+color.BLUE + "** " + text.upper() + " **" + color.END+'\n')


def ptitl(text):
    print('\n'+color.BOLD+color.UNDERLINE+color.BLUE + "** " + text.upper() + " **" + color.END+'\n')