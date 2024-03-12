from typing import Any, Optional, TextIO, Union


FileDict = dict[str, Optional[Union[str, TextIO]]]
FilenameTuple = tuple[Optional[str], str]


def define_file_kwargs(kwargs: dict[str, Any]) -> FileDict:
    current_file_kwargs: FileDict = {
        "results_location": kwargs["results_location"],
        "training_file": None,
        "evaluation_file": None,
        "model_location": None,
        "model_output_location": None,
    }

    if kwargs.get("results_location", None) is not None:
        output_filenames: list[FilenameTuple] = [
            (kwargs.get("training_filename", None), "training_file"),
            (kwargs.get("evaluation_filename", None), "evaluation_file"),
        ]

        for (filename, file_type) in output_filenames:
            if filename is not None:
                new_filepath: str = f"{kwargs['results_location']}/{filename}"
                new_filepath += ".txt"
                current_file_kwargs[file_type] = open(new_filepath, encoding="utf-8", mode="w+")

    if kwargs["mode"] in ("evaluate", "predict") and (not kwargs["model_location"] or not kwargs["model_name"]):
        raise ValueError(f"A filename must be supplied for the <{kwargs['mode']}> mode. Please try again.")
    elif kwargs["model_location"] and kwargs["model_name"]:
        model_filepath: str = f"{kwargs['model_location']}/{kwargs['model_name']}.pt"
        model_outputs_filepath: str = f"{kwargs['model_location']}/{kwargs['model_name']}.json"
        current_file_kwargs["model_location"] = model_filepath
        current_file_kwargs["model_output_location"] = model_outputs_filepath

    return current_file_kwargs
