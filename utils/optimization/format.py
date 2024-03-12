from typing import Any, Optional


BASH_FORMAT: str = "#!/bin/bash\n" \
                   "# Produced with random seed {0}...\n\n" \
                   "# Main Commands:\n" \
                   "conda activate EvaLatin-2024\n" \
                   "{1}"


class CommandTemplate:
    def __init__(self, script: str):
        self.script = script

    def __call__(self, mode: str, agnostic_options: Optional[dict[str, Any]] = None,
                 common_options: Optional[dict[str, Any]] = None, mode_options: Optional[dict[str, Any]] = None):
        command_segments: list[str] = [f"python3 {self.script} {mode}"]
        for options in (agnostic_options, common_options, mode_options):
            if options is not None:
                formatted_option_string: str = self._format_options(options)
                command_segments.append(formatted_option_string)

        command: str = " ".join(command_segments)
        return command

    @staticmethod
    def _format_options(options: dict[str, Any]) -> str:
        formatted_options: list[str] = []
        for option_name, option_value in options.items():
            if isinstance(option_value, bool):
                if option_value is True:
                    formatted_option: str = f"--{option_name.replace('_', '-')}"
                else:   # option_value is False
                    formatted_option: str = f"--no-{option_name.replace('_', '-')}"
            else:
                formatted_option_name: str = f"--{option_name.replace('_', '-')}"
                formatted_option: str = f"{formatted_option_name} {option_value}"
            formatted_options.append(formatted_option)

        formatted_option_string: str = " ".join(formatted_options)
        return formatted_option_string


def get_numeric_string(number: int, places: int):
    numeric_string: str = str(number)
    while len(numeric_string) < places:
        numeric_string = "0" + numeric_string
    return numeric_string
