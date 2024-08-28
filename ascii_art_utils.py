import pyfiglet
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

def create_ascii_art(text, font='standard'):
    """
    Create ASCII art from the given text using the specified font.

    :param text: The text to be converted into ASCII art.
    :param font: The font to be used for the ASCII art. Default is 'standard'.
    :return: A string containing the ASCII art representation of the text.
    """
    # Create an instance of the Figlet class with a specific font
    figlet = pyfiglet.Figlet(font=font)
    # Generate the ASCII art for the provided text
    ascii_art = figlet.renderText(text)
    return ascii_art

def colorize_text(text, fg_color, bold=False):
    """
    Apply color and optional bold formatting to the provided text.

    :param text: The text to be colorized.
    :param fg_color: The foreground color name. Must be one of 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    :param bold: If True, apply bold formatting to the text. Default is False.
    :return: The colorized and optionally bold text.
    """
    # Define color codes using colorama
    fg_colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
    }
    
    # Construct the color string
    color_string = fg_colors.get(fg_color, Fore.WHITE) 
    if bold:
        color_string += Style.BRIGHT
    # Return the text colored according to the specified colors
    return color_string + text + Style.RESET_ALL


