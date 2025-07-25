# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import glob
from tests.utils import comp_pcc
from tools.data_collection import pydantic_models
from enum import Enum
import string
import warnings

# Map dictionary keys from metrics to header descriptions
csv_header_mappings = {
    "model": ("Model", "Name of the model."),
    "success": (
        "Status",
        """Indicates whether the model is:
- ✅ End-to-end on device: All PyTorch operations have been converted to TT-NN operations.
- 🚧 Compiled: The converted model runs but some operations still fallback to PyTorch. This may be due to an unsupported operation or configuration.
- ❌ Traced: The model does not run but its PyTorch operations are traced for future development. This may indicate a temporary incompatibility with a compiler pass.""",
    ),
    "batch_size": (
        "Batch",
        "Batch size used for inference",
    ),
    "compiled_first_run": (
        "Compiled First Run (ms)",
        "Time until the first compiled run finishes (ms), including compilation time and warming caches.",
    ),
    "original_throughput": (
        "Original Throughput (Inferences Per Second)",
        "Execution throughput (in inferences per second) of the model before conversion.",
    ),
    "compiled_throughput": (
        "Compiled Throughput (Inferences Per Second)",
        "Execution throughput (in inferences per second) of the model after conversion, once caches are warm.",
    ),
    "accuracy": (
        "Accuracy (%)",
        "Model accuracy on a predefined test dataset after conversion.",
    ),
    "torch_ops_total_unique_before": (
        "Torch Ops Before (Unique Ops)",
        "The total number of operations used by the model in the original Torch implementation. The number in parentheses represents the total unique ops.",
    ),
    "torch_ops_total_unique_remain": (
        "Torch Ops Remain (Unique Ops)",
        "The total number of operations used after conversion to TT-NN. The number in parentheses represents the total unique ops.",
    ),
    "to_from_device_ops": (
        "To/From Device Ops",
        "The number of `to/from_device` operations (data transfer to/from the device).",
    ),
}


# Load a pickle file from path and return an object or None
def load_pickle(path: str):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def load_single_metrics(path: str):
    if os.path.isdir(path):
        results = []
        for file in os.listdir(path):
            if file.endswith(".pickle"):
                result = load_pickle(os.path.join(path, file))
                if result:
                    results += result
        return results
    else:
        return []


# Load a pt file from path and return a Torch tensor object or None
def load_pt(path: str):
    if os.path.isfile(path):
        return torch.load(path)
    else:
        return None


def create_aten_op_dict(aten_ops_before_list, aten_ops_remain_list):
    """Take a before and after list of ops and return a table of information

    Returns:
        Dictionary with the keys: ["aten ops", "status", "count"]
    """
    original_aten_ops_set = set(aten_ops_before_list)
    remaining_aten_ops_set = original_aten_ops_set.intersection(aten_ops_remain_list)

    original_aten_ops_count = Counter(aten_ops_before_list)
    aten_ops_dict = {"aten ops": [], "status": [], "count": []}
    for op in sorted(list(original_aten_ops_set)):
        aten_ops_dict["aten ops"].append(op)
        aten_ops_dict["count"].append(original_aten_ops_count[op])
        if op in remaining_aten_ops_set:
            aten_ops_dict["status"].append("✘")
        else:
            aten_ops_dict["status"].append("✅")
    return aten_ops_dict


def write_to_readme(all_metrics, aten_ops_per_model):
    """Write collected metrics to sections of the README.

    Current process:
    * Writes a table that contains a summary of currently tested models.
    * Writes a series of tables of each model containing the statuses of ops used.
    """
    # Create an explanation section for the headers
    explanations_md = "\n".join([f"**{val[0]}**: {val[1]}  " for val in csv_header_mappings.values()])

    # Load README.in as an f-string and substitute the variables
    with open("docs/README.md.in", "r") as text_file:
        readme_in = text_file.read()

    readme_comment = (
        "[comment]: <> (This README.md was generated by tools/collect_metrics.py.)\n"
        "[comment]: <> (Please modify docs/README.md.in and/or collect_metrics.py to make permanent changes.)\n"
    )

    # Convert metrics to markdown table
    all_metrics_sort = (
        [m for m in all_metrics if m["Status"] == "✅"]
        + [m for m in all_metrics if m["Status"] == "🚧"]
        + [m for m in all_metrics if m["Status"] == "❌"]
    )
    metrics_md = pd.DataFrame(all_metrics_sort).to_markdown(index=False)

    # Write to README file
    readme_md = readme_comment + readme_in.format(
        metrics_md=metrics_md,
        explanations_md=explanations_md,
    )
    with open("README.md", "w") as text_file:
        print(readme_md, file=text_file)
    print("Data written to README.md")


# This class labels the conversion status of an Op
class ConversionStatus(Enum):
    DONE = (1,)  # Conversion is successful
    FALLBACK = (2,)  # Only some are converted successfully
    BUG = (3,)  # Known issue with conversion
    NONE = (4,)  # No conversion at all
    UNKNOWN = (5,)  # Op was not processed, so status is unknown
    REMOVED = (6,)  # Op is removed from compiled graph; usually intentional


class InputVarPerOp(defaultdict):
    """
    2-D nested dict where `["opname"]["input_variation"] = status`
    """

    class InputVarStatus(defaultdict):
        """
        Holds the status of each input variation `["input_variation"] = status`
        """

        def __init__(self):
            super(InputVarPerOp.InputVarStatus, self).__init__(lambda: ConversionStatus.UNKNOWN)

        def get_completion_status_for(self) -> str:
            """
            Return a "✅" if the sum of ConversionStatus.DONE and ConversionStatus.REMOVED is equal the total.
            Return a "🚧" if the sum of conversions are greater than 0, but less than the total.
            Otherwise return an "✘".}
            """
            done = self.get_conversion_status_count_for(ConversionStatus.DONE)
            removed = self.get_conversion_status_count_for(ConversionStatus.REMOVED)
            completed = done + removed
            total = len(self.values())
            assert completed <= total, f"done: {done} + removed: {removed} ({completed}) is greater than total: {total}"
            if completed == total:
                return "✅"
            elif completed < total and completed > 0:
                return "🚧"
            else:
                return "✘"

        def get_conversion_status_count_for(self, status: ConversionStatus) -> int:
            """
            Get the count of a type of ConversionStatus
            """
            count = 0
            for input in self.values():
                if input == status:
                    count += 1
            return count

        def get_generality_score(self) -> float:
            """
            Generality Score: Converted input variations / Total input variations
            """
            done = self.get_conversion_status_count_for(ConversionStatus.DONE)
            removed = self.get_conversion_status_count_for(ConversionStatus.REMOVED)
            completed = done + removed
            total = len(self.values())
            return round(completed / total, 2)

        def get_list_input_var_to_dict(self):
            """
            Convert list of InputVariations to markdown-compatible dict
            """
            sort_by_opname = dict(sorted(self.items()))
            return {
                "ATen Input Variations": [x for x in sort_by_opname.keys()],
                "Status": [string.capwords(x.name) for x in sort_by_opname.values()],
            }

    def __init__(
        self,
        original_schema_metrics={},
        compiled_schema_metrics={},
        single_metrics=[],
        compiled_run_success: bool = False,
    ):
        super(InputVarPerOp, self).__init__(self.InputVarStatus)
        self.ops_dir = "operations"
        self.compiled_run_success = compiled_run_success

        def _join_br(str_list: list):
            # Separate each input with a line-break, <br>
            return ",<br>".join(str_list)

        if original_schema_metrics:
            # classify each input variation with the op
            for op in original_schema_metrics:
                opname = op["opname"]
                inputs = _join_br(op["inputs"])
                self[opname][inputs]
                if opname == "aten.cat.default":
                    print(op)
            # If exist, map converted ops to the original op
            if compiled_schema_metrics:
                # Hold ops that require revisiting the original dict to determine the status
                unprocessed_compiled_ops = InputVarPerOp()
                for op in compiled_schema_metrics:
                    if "original_inputs" in op:
                        opname = op["opname"]
                        original_opname = op["original_inputs"]["opname"]
                        original_inputs = _join_br(op["original_inputs"]["inputs"])
                        if opname == "aten.cat.default":
                            print(op)
                        # NOTE(kevinwuTT): Some ttnn ops are wrapped, so they have no `ttnn` prefix. Should this be more strict?
                        if opname != original_opname:
                            # Some aten ops are converted to other aten ops
                            if opname.startswith("aten"):
                                self[original_opname][original_inputs] = ConversionStatus.FALLBACK
                            else:
                                self[original_opname][original_inputs] = ConversionStatus.DONE
                        else:
                            unprocessed_compiled_ops[opname][original_inputs]
                    else:
                        warnings.warn(
                            f'{op} has no "original_inputs" key for compiled schema metrics. This may indicate a newly inserted op.'
                        )

                # Process remaining ops
                for opname, input_vars in self.items():
                    for input_var, status in input_vars.items():
                        # If opname and the same input variation still exists, then this op has no conversion
                        unprocessed_compiled_ops_input_vars = unprocessed_compiled_ops.get(opname)
                        if (
                            unprocessed_compiled_ops_input_vars != None
                            and unprocessed_compiled_ops_input_vars.get(input_var) != None
                        ):
                            self[opname][input_var] = ConversionStatus.NONE
                        # If opname does not exist, or it exists, but matching input variation does not exist,
                        # and the status is unknown and the compiled run was successful, then this op is considered removed
                        elif self[opname][input_var] == ConversionStatus.UNKNOWN and self.compiled_run_success:
                            self[opname][input_var] = ConversionStatus.REMOVED

                # Sanity check
                for opname, input_vars in self.items():
                    if input_vars.get_conversion_status_count_for(ConversionStatus.UNKNOWN) != 0:
                        warnings.warn(f"{opname}: {input_vars} has UNKNOWN status.")

        elif compiled_schema_metrics:
            # if only compiled_schema exists, initialize dict with those values
            for op in compiled_schema_metrics:
                opname = op["opname"]
                # FIXME: input variations are not saved for ttnn ops, saving aten schema instead
                if "original_inputs" in op:
                    inputs = _join_br(op["original_inputs"]["inputs"])
                else:
                    inputs = _join_br(op["inputs"])
                self[opname][inputs]
        # else this will be an empty dict with defaults

        self.single_metrics = single_metrics
        if type(self.single_metrics) == list and len(self.single_metrics) > 0:
            for s in self.single_metrics:
                s["input_variation"] = _join_br(s["input_strings"])

    def merge(self, other: "InputVarPerOp"):
        """
        Method to merge another InputVarPerOp object to this.
        """
        for opname, input_var in other.items():
            for input, status in input_var.items():
                # Don't overwrite existing with UNKNOWN status
                if self[opname][input] == ConversionStatus.UNKNOWN:
                    self[opname][input] = status
        if not self.single_metrics:
            self.single_metrics = other.single_metrics
        elif self.single_metrics:
            self.single_metrics += other.single_metrics

    def sorted(self):
        return dict(sorted(self.items()))

    def append_single_status(self, opname, input_variations, input_vars_dict):
        """
        Add the single status of each input variation.
        """

        def _filter_single_status(opname, input_variation):
            """
            Filter out the single status from single_metrics.
            """
            na = {
                "native_run": "N/A",
                "run": "N/A",
                "accuracy": "N/A",
                "convert_to_ttnn": "N/A",
                "ttnn_fallbacks_to_host_count": "N/A",
            }
            if not self.single_metrics:
                return na
            return next(
                filter(
                    lambda x: x["opname"] == opname and x["input_variation"] == input_variation, self.single_metrics
                ),
                na,
            )

        input_vars_dict["Isolated"] = []
        input_vars_dict["PCC"] = []
        input_vars_dict["Host"] = []
        for input_variation in input_variations:
            single_status = _filter_single_status(opname, input_variation)
            status = "None"
            # aten ir should run success, or the single op testcase is illegal
            if single_status["native_run"] != True:
                status = "Unknown"
            # if compiled graph run fail, then the status is failed
            elif single_status["run"] == False:
                status = "Failed"
            # or status is done or fallback according to the convert_to_ttnn
            elif single_status["run"] == True and single_status["convert_to_ttnn"] == True:
                status = "Done"
            elif single_status["run"] == True and single_status["convert_to_ttnn"] == False:
                status = "Fallback"
            input_vars_dict["Isolated"].append(status)
            input_vars_dict["PCC"].append(single_status["accuracy"])
            input_vars_dict["Host"].append(single_status["ttnn_fallbacks_to_host_count"])

    def generate_md_for_input_variations(self) -> str:
        """
        Convert current dict to a markdown string that consists of a high level table of
        each op and individual tables of each op and their input variation statuses.

        Returns:
            Markdown string
        """
        md = self.generate_md_for_high_level_table()
        md += "***\n"

        for opname, input_vars in self.sorted().items():
            input_vars_dict = input_vars.get_list_input_var_to_dict()
            self.append_single_status(opname, input_vars_dict["ATen Input Variations"], input_vars_dict)
            md += f"### {opname}\n"
            md += pd.DataFrame(input_vars_dict).to_markdown(index=True) + "\n"

        return md

    def generate_md_for_high_level_table(self, links_to_ops=False):
        # Create a high level table for the main page
        high_level_op_status = defaultdict(list)
        sort_by_opname = self.sorted()
        for opname, input_vars in sort_by_opname.items():
            ops = f"[{opname}]({self.ops_dir}/{opname}.md)" if links_to_ops else opname
            high_level_op_status["Operations"].append(ops)
            high_level_op_status["Input Variations"].append(len(input_vars))
            high_level_op_status["Converted"].append(input_vars.get_conversion_status_count_for(ConversionStatus.DONE))
            high_level_op_status["Removed"].append(input_vars.get_conversion_status_count_for(ConversionStatus.REMOVED))
            high_level_op_status["Fallback"].append(
                input_vars.get_conversion_status_count_for(ConversionStatus.FALLBACK)
            )
            high_level_op_status["Completed"].append(input_vars.get_completion_status_for())
            high_level_op_status["Score"].append(input_vars.get_generality_score())

        md = ""
        md += f"# High Level Operations Status\n"
        md += pd.DataFrame(high_level_op_status).to_markdown(index=True) + "\n"

        return md

    def write_md(self, md: str, basedir: Path, filename: Path):
        basedir.mkdir(parents=True, exist_ok=True)
        with open(basedir / filename, "w") as text_file:
            print(md, file=text_file)
        print(f"Data written to {basedir / filename}")

    def write_md_for_cumulative_report(self):
        md = self.generate_md_for_high_level_table(links_to_ops=True)

        basedir = Path("docs")
        self.write_md(md, basedir, Path("OperationsReport.md"))

        # Create a new page for each op
        cumulative_ops_dir = Path(self.ops_dir)
        cumulative_ops_dir.mkdir(parents=True, exist_ok=True)

        for opname, input_vars in self.sorted().items():
            op_md = ""
            input_vars_dict = input_vars.get_list_input_var_to_dict()
            self.append_single_status(opname, input_vars_dict["ATen Input Variations"], input_vars_dict)
            op_md += f"### {opname}\n"
            op_md += pd.DataFrame(input_vars_dict).to_markdown(index=True) + "\n"
            self.write_md(op_md, basedir / cumulative_ops_dir, Path(f"{opname}.md"))

    def write_md_for_input_variations(self, basedir: Path, filename: Path):
        """
        Convert to a markdown string and write to a file.
        """
        md = self.generate_md_for_input_variations()
        self.write_md(md, basedir, filename)

    def serialize_to_pydantic_operations(self):
        """Transform schema information to a list of `class Operation` pydantic models."""
        operations = []
        for opname, inputs in self.items():
            for input_var in inputs.keys():
                operations.append(pydantic_models.Operation(op_name=opname, op_schema=input_var))
        return operations


if __name__ == "__main__":
    # Holds the concatenation of all the metrics for each model
    all_metrics = []

    cumulative_input_vars = InputVarPerOp()

    # Hold aten ops per model
    aten_ops_per_model = {}

    # Assumed directory structure example. Some files will not exist if test failed.
    """
    pytorch2.0_ttnn
    ├── metrics
        ├── BERT
        │   ├── compiled-run_time_metrics.pickle
        │   ├── compiled-schema_list.pickle
        │   ├── original-runtime_metrics.pickle
        │   └── original-schema_list.pickle
        └── ResNet18
            ├── compiled-run_time_metrics.pickle
            └── compiled-schema_list.pickle
    """
    if not os.path.isdir("metrics"):
        raise ValueError("metrics directory not found. Please run models to generate metrics first.")

    # Support subdirectories
    all_model_paths = sorted([Path(dirpath) for dirpath, dirnames, filenames in os.walk("metrics") if not dirnames])

    for model_path in all_model_paths:
        # Remove the "metrics" root directory and convert to string
        model = str(Path(*model_path.parts[1:]))

        # Load run time metrics
        original_runtime_metrics_path = model_path / "original-run_time_metrics.pickle"
        compiled_runtime_metrics_path = model_path / "compiled-run_time_metrics.pickle"
        original_runtime_metrics = load_pickle(original_runtime_metrics_path)
        compiled_runtime_metrics = load_pickle(compiled_runtime_metrics_path)
        # Both run time files should exist
        assert original_runtime_metrics, f"{original_runtime_metrics_path} file not found"
        assert compiled_runtime_metrics, f"{compiled_runtime_metrics_path} file not found"

        # Initialize the Pydantic model
        pydantic_model = pydantic_models.ModelRun(name=model)

        batch_size = compiled_runtime_metrics.get("batch_size", 1)

        # Rename run_time keys to distinguish between original or compiled
        pydantic_model.batch_size = batch_size
        if "run_time" in original_runtime_metrics:
            run_time = original_runtime_metrics.pop("run_time")
            original_runtime_metrics["original_throughput"] = batch_size / run_time * 1000
            pydantic_model.original_run_time = float(run_time)
        if "run_time" in compiled_runtime_metrics:
            run_time = compiled_runtime_metrics.pop("run_time")
            compiled_runtime_metrics["compiled_throughput"] = batch_size / run_time * 1000
            pydantic_model.compiled_run_time = float(run_time)
            compiled_runtime_metrics["compiled_first_run"] = compiled_runtime_metrics["run_time_first_iter"]

        # Replace compiled runtime with average if available
        if "avg_run_time" in compiled_runtime_metrics:
            avg_run_time = compiled_runtime_metrics.pop("avg_run_time")
            compiled_runtime_metrics["compiled_run_time"] = avg_run_time
            pydantic_model.compiled_run_time = float(avg_run_time)

        # Load op schema metrics
        original_schema_metrics_path = model_path / "original-schema_list.pickle"
        original_schema_metrics = load_pickle(original_schema_metrics_path) or {}

        compiled_schema_metrics_path = model_path / "compiled-schema_list.pickle"
        compiled_schema_metrics = load_pickle(compiled_schema_metrics_path) or {}

        # Load single metrics
        single_metrics_path = Path("metrics-autogen-op") / model
        single_metrics = load_single_metrics(single_metrics_path)

        # Count total number of original aten ops and unique aten ops
        ops_metrics = {
            "torch_ops_total_unique_before": "N/A",
            "torch_ops_total_unique_remain": "N/A",
            "to_from_device_ops": "N/A",
        }
        if original_schema_metrics:
            aten_ops_before_list = [
                node["opname"] for node in original_schema_metrics if node["opname"].startswith("aten.")
            ]
            aten_ops_before, aten_ops_unique_before = len(aten_ops_before_list), len(set(aten_ops_before_list))
            ops_metrics["torch_ops_total_unique_before"] = f"{aten_ops_before} ({aten_ops_unique_before})"

            # Count number of aten ops remaning after conversion. Only relevant if model ran successfully.
            if compiled_schema_metrics and compiled_runtime_metrics["success"]:
                aten_ops_remain_list = [
                    node["opname"] for node in compiled_schema_metrics if node["opname"].startswith("aten.")
                ]
                aten_ops_remain, aten_ops_unique_remain = len(aten_ops_remain_list), len(set(aten_ops_remain_list))
                ops_metrics["torch_ops_total_unique_remain"] = f"{aten_ops_remain} ({aten_ops_unique_remain})"

                device_op_list = [
                    node["opname"]
                    for node in compiled_schema_metrics
                    if node["opname"].startswith("ttnn.to") or node["opname"].startswith("ttnn.from")
                ]
                ops_metrics["to_from_device_ops"] = f"{len(device_op_list)}"

                # Compile list of aten ops per model used for README
                aten_ops_per_model[model] = create_aten_op_dict(aten_ops_before_list, aten_ops_remain_list)

        # Get accuracy metric. Will not exist if test failed.
        if "accuracy" in compiled_runtime_metrics and compiled_runtime_metrics["accuracy"] is not None:
            acc = round(compiled_runtime_metrics["accuracy"] * 100, 2)
            accuracy_metric = {"accuracy": acc}
            pydantic_model.accuracy = acc
        else:
            accuracy_metric = {"accuracy": "N/A"}

        # Save run_success status before changing it
        compiled_run_success = compiled_runtime_metrics["success"]
        pydantic_model.run_success = compiled_run_success
        # Remap bool to emoji
        if compiled_run_success and aten_ops_remain == 0:
            compiled_runtime_metrics["success"] = "✅"
        else:
            emoji_map = {True: "🚧", False: "❌"}
            compiled_runtime_metrics["success"] = emoji_map[compiled_run_success]

        # Process input variations per model
        input_var_per_op = InputVarPerOp(
            original_schema_metrics, compiled_schema_metrics, single_metrics, compiled_run_success=compiled_run_success
        )
        model_info_dir = Path("docs") / Path("models") / Path(model)
        input_var_per_op.write_md_for_input_variations(model_info_dir, Path("input_variations.md"))

        # Populate schemas for each op for original graph
        pydantic_model.ops_original = input_var_per_op.serialize_to_pydantic_operations()

        # Populate schemas for each op for compiled graph
        compiled_input_var_per_op = InputVarPerOp(compiled_schema_metrics=compiled_schema_metrics)
        pydantic_model.ops_compiled = compiled_input_var_per_op.serialize_to_pydantic_operations()

        # Add links that point to the directory of the model info
        model_metric = {"model": f"[{model}](<{model_info_dir}>)"}
        pydantic_model.path_in_repo = str(model_info_dir)

        # Collect cumulative input variations
        cumulative_input_vars.merge(input_var_per_op)

        # Concatenate all the metrics together
        cat_metrics = {
            **original_runtime_metrics,
            **compiled_runtime_metrics,
            **ops_metrics,
            **accuracy_metric,
            **model_metric,
        }
        # Remap original keys with header descriptions to prepare for markdown table
        cat_metrics_remapped = {}
        for key, val in csv_header_mappings.items():
            if key in cat_metrics:
                cat_metrics_remapped[val[0]] = cat_metrics[key]
            else:
                cat_metrics_remapped[val[0]] = "N/A"

        all_metrics.append(cat_metrics_remapped)

        # Links to graphs
        if os.path.isfile(f"metrics/{model}/00.origin.dot.svg"):
            pydantic_model.graph_before = f"metrics/{model}/00.origin.dot.svg"
        # Search for the last graph available
        svg_list = sorted(glob.glob(f"metrics/{model}/*.svg"))
        if len(svg_list) > 1:
            pydantic_model.graph_after = svg_list[-1]

        # Generate JSON using Pydantic
        model_run_json = pydantic_model.model_dump_json(indent=2)
        model_run_filename = f"metrics/{model}/model_run.json"
        with open(model_run_filename, "w") as text_file:
            print(model_run_json, file=text_file)

        print(f"Data written to {model_run_filename}")

    # Write cumulative input variations to file
    cumulative_input_vars.write_md_for_cumulative_report()

    # Write collected metrics to README
    write_to_readme(all_metrics, aten_ops_per_model)
