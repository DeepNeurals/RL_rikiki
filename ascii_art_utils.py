import pyfiglet
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

def create_ascii_art(text, font='standard'):
    # Create an instance of the Figlet class with a specific font
    figlet = pyfiglet.Figlet(font=font)
    # Generate the ASCII art for the provided text
    ascii_art = figlet.renderText(text)
    return ascii_art

def colorize_text(text, fg_color, bold=False):
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


