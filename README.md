# ![alt text](https://github.com/DeepNeurals/RL_rikiki/blob/main/images/rikiki.png)

This repository contains the implementation of an AI agent that plays Rikiki, a card game, using Reinforcement Learning techniques. You can play against the AI agent manually or let the AI play automatically.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Manual Play](#manual-play)
- [Automatic Play](#automatic-play)

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone git@github.com:DeepNeurals/RL_rikiki.git
    ```

2. Navigate to the project directory:

    ```bash
    cd RL_rikiki
    ```
## Setup

To install all necessary dependencies, run the provided `setup.sh` script. This script installs all required Python packages listed in the `requirements.txt` file.

1. Make the script executable (only needed once):

    ```bash
    chmod +x setup.sh
    ```

2. Run the setup script:

    ```bash
    ./setup.sh
    ```

This will install all required dependencies, including `torch`, `pydealer`, `tabulate`, `pyfiglet`, and `colorama`.

## Usage

To run the Rikiki game, you need to execute the `minimal_version.py` file. By default, the game runs in automatic mode where the AI plays without manual input.

### Manual Play

If you want to play Rikiki manually against the AI agent, follow these steps:

1. Open the `minimal_version.py` file in your preferred text editor.
2. Locate the `manual_input` variable (near the bottom of the file) and set it to `True`:

    ```python
    manual_input = True
    ```

3. Save the file and return to your terminal.
4. Run the game with the following command:

    ```bash
    python3 minimal_version.py
    ```

### Automatic Play

To let the AI play automatically without manual input, ensure the `manual_input` variable is set to `False` (which is the default):

```python
manual_input = False
```


