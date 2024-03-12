from argparse import ArgumentParser, Namespace
from random import choice, seed, randint
from sys import maxsize
from typing import Any, Optional, TextIO

from utils.cli.messages import GeneralMessage, HyperparameterMessage
from utils.optimization.format import get_numeric_string, CommandTemplate, BASH_FORMAT
from utils.optimization.hyperparameter import HYPERPARAMETERS, HyperparameterParseAction, handle_training_constraints

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--command-filepath", type=str, help=HyperparameterMessage.COMMAND_FILEPATH)
    parser.add_argument(
        "--command-format", type=str, choices=["text", "bash"], default="text",
        help=HyperparameterMessage.COMMAND_FORMAT
    )
    parser.add_argument("--model-location", type=str, default="models", help=GeneralMessage.MODEL_LOCATION)
    parser.add_argument("--model-name", type=str, default="model", help=GeneralMessage.MODEL_NAME)
    parser.add_argument("--results-location", type=str, default="results", help=GeneralMessage.RESULTS_LOCATION)
    parser.add_argument("--training-filename", type=str, default="training", help=GeneralMessage.TRAINING_FILENAME)
    parser.add_argument(
        "--validation-filename", type=str, default="validation", help=HyperparameterMessage.VALIDATION_FILENAME
    )
    parser.add_argument("--test-filename", type=str, default="test", help=HyperparameterMessage.TEST_FILENAME)
    parser.add_argument("--training-interval", type=str, default="inf", help=GeneralMessage.TRAINING_INTERVAL)
    parser.add_argument(
        "--inference-split", type=str, choices=["train", "validation", "test"], default="test",
        help=GeneralMessage.INFERENCE_SPLIT
    )
    parser.add_argument("--seed", type=int, default=randint(0, maxsize), help=GeneralMessage.RANDOM_SEED)
    parser.add_argument(
        "--specified", action=HyperparameterParseAction, nargs="*", default={}, help=HyperparameterMessage.SPECIFIED
    )
    parser.add_argument("--trials", type=int, default=16, help=HyperparameterMessage.TRIALS)
    parser.add_argument("--trial-start-offset", type=int, default=0, help=HyperparameterMessage.TRIAL_OFFSET)

    parser.add_argument(
        "--varied", type=str, nargs="*", choices=tuple(HYPERPARAMETERS.keys()), help=HyperparameterMessage.VARIED
    )
    args: Namespace = parser.parse_args()
    seed(args.seed)

    template: CommandTemplate = CommandTemplate("polarity_detector.py")

    current_trial: int = 1
    trial_hyperparameters: list[dict[str, Any]] = []
    trial_training_commands: list[str] = []
    trial_evaluation_commands: list[str] = []
    while current_trial <= args.trials:
        agnostic_hyperparameters: dict[str, Any] = {}

        numeric_string_segment: str = f"{get_numeric_string(current_trial + args.trial_start_offset, 3)}"
        agnostic_hyperparameters["model_location"] = args.model_location
        agnostic_hyperparameters["model_name"] = f"{args.model_name}_{numeric_string_segment}"
        agnostic_hyperparameters["results_location"] = args.results_location

        training_filename: str = f"{args.training_filename}_{numeric_string_segment}"
        validation_filename: str = f"{args.validation_filename}_{numeric_string_segment}"
        test_filename: str = f"{args.test_filename}_{numeric_string_segment}"

        common_hyperparameters: dict[str, Any] = {}
        training_hyperparameters: dict[str, Any] = {
            "training_filename": training_filename,
            "evaluation_filename": validation_filename,
            "training_interval": args.training_interval,
        }
        evaluation_hyperparameters: dict[str, Any] = {
            "evaluation_filename": test_filename,
            "inference_split": args.inference_split
        }
        for hyperparameter in HYPERPARAMETERS.values():
            if hyperparameter.mode == "common":
                hyperparameter_collection: dict[str, Any] = common_hyperparameters
            elif hyperparameter.mode == "training":
                hyperparameter_collection: dict[str, Any] = training_hyperparameters
            else:
                raise ValueError(f"The mode <{hyperparameter.mode}> is currently not supported.")

            if args.specified is not None and hyperparameter.name in args.specified.keys():
                hyperparameter_collection[hyperparameter.name] = args.specified[hyperparameter.name]
            elif args.varied is not None and hyperparameter.name in args.varied:
                hyperparameter_collection[hyperparameter.name] = choice(hyperparameter.range)
            else:
                hyperparameter_collection[hyperparameter.name] = hyperparameter.default

        # We get rid of any unnecessary hyperparameters generated.
        handle_training_constraints(common_hyperparameters, training_hyperparameters)

        all_pairs: list[tuple[str, Any]] = [*common_hyperparameters.items(), *training_hyperparameters.items()]
        full_hyperparameters: dict[str, Any] = {key: value for (key, value) in all_pairs}
        if full_hyperparameters not in trial_hyperparameters:
            trial_hyperparameters.append(full_hyperparameters)

            training_command: str = template(
                "train", agnostic_options=agnostic_hyperparameters,
                common_options=common_hyperparameters, mode_options=training_hyperparameters
            )
            evaluation_command: str = template(
                "evaluate", agnostic_options=agnostic_hyperparameters,
                common_options=common_hyperparameters, mode_options=evaluation_hyperparameters
            )

            if args.command_filepath is not None:
                trial_training_commands.append(f"{training_command}\n")
                trial_evaluation_commands.append(f"{evaluation_command}\n")
            else:
                print(training_command)
                print(evaluation_command)

            current_trial += 1

    if args.command_filepath is not None:
        if args.command_format == "text":
            output_filepath: str = f"{args.command_filepath}.txt"
            output_file: Optional[TextIO] = open(output_filepath, encoding="utf-8", mode="w+")
            for i in range(0, len(trial_training_commands)):
                output_file.write(trial_training_commands[i])
                output_file.write(trial_evaluation_commands[i])

            output_file.close()
        elif args.command_format == "bash":
            current_bash_trial: int = 1
            while current_bash_trial <= args.trials:
                numeric_string_segment: str = get_numeric_string(current_bash_trial + args.trial_start_offset, 3)
                bash_filename: str = f"{args.command_filepath}_{numeric_string_segment}.sh"
                bash_file: TextIO = open(bash_filename, encoding="utf-8", mode="w+")
                bash_file.write(BASH_FORMAT.format(args.seed, trial_training_commands[current_bash_trial - 1]))
                bash_file.close()

                bash_eval_filename: str = f"{args.command_filepath}_eval_{numeric_string_segment}.sh"
                bash_eval_file: TextIO = open(bash_eval_filename, encoding="utf-8", mode="w+")
                bash_eval_file.write(BASH_FORMAT.format(args.seed, trial_evaluation_commands[current_bash_trial - 1]))
                bash_eval_file.close()

                current_bash_trial += 1
        else:
            raise ValueError(f"Output format <{args.command_format}> not recognized.")
