import argparse
from collections import defaultdict
from pathlib import Path

import natsort
import numpy as np
from natsort import natsorted
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm


def get_metrics(df, score):
    score_arr = df[score].values
    score_arr[(score_arr == np.inf) ] = score_arr[score_arr != np.inf].max() + 1
    score_arr[np.isnan(score_arr)] = score_arr[~np.isnan(score_arr)].max() + 1
    result = {
        # "mean_rank": rankdata(-score_arr)[df["has_problem"]].mean(),
        "ap": average_precision_score(df["has_problem"], score_arr) * 100,
    }

    return result


def get_results_rows(df):
    ordered_columns = ['random', 'ppl', 'p_mean', 'p_min', 'aum']
    df_to_save = df.pivot_table(index='score', columns='problem', values='ap')
    df_to_save = df_to_save.reindex(ordered_columns)
    problem_to_scores = {}
    for problem in df_to_save.keys().unique():
        problem_to_scores[problem] = df_to_save[problem].values

    return problem_to_scores



def build_results_df(df):
    results_df = defaultdict(list)
    for problem in sorted(df["problem"].unique()) + ["all"]:
        if problem.startswith("none") or problem == "unknown":
            continue

        if problem != "all":
            df_problem = df[(df["problem"] == problem) | (df["problem"] == "none_" + problem)].copy()
        else:
            df_problem = df[(df["problem"] != "unknown")].copy()

        df_problem["has_problem"] = df_problem["has_problem"] > 0

        results_df["ap"].append(df_problem["has_problem"].mean() * 100)
        results_df["problem"].append(problem)
        results_df["score"].append("random")

        results = get_metrics(df_problem, "ppl")
        for k, v in results.items():
            results_df[k].append(v)
        results_df["problem"].append(problem)
        results_df["score"].append("ppl")

        results = get_metrics(df_problem, "aum")
        for k, v in results.items():
            results_df[k].append(v)
        results_df["problem"].append(problem)
        results_df["score"].append("aum")

        results = get_metrics(df_problem, "p_mean")
        for k, v in results.items():
            results_df[k].append(v)
        results_df["problem"].append(problem)
        results_df["score"].append("p_mean")

        results = get_metrics(df_problem, "p_min")
        for k, v in results.items():
            results_df[k].append(v)
        results_df["problem"].append(problem)
        results_df["score"].append("p_min")

    return pd.DataFrame(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--adapted", action="store_true")
    parser.add_argument("--task_agg", default=None)
    args = parser.parse_args()

    if args.adapted:
        scores_dir = Path("adapted_scores")
    else:
        scores_dir = Path("scores")

    size_to_lines = {"small": [], "base": [], "large": [], "xl": []}
    sizes = ["small", "base", "large", "xl"]
    

    seeds = [42, 43, 44]
    for size in tqdm(list(size_to_lines)):
        results_rows = defaultdict(list)
        for seed in tqdm(seeds):
            files = natsorted(scores_dir.glob(f"p4_runs_large_{args.data}_{seed}_t5-{size}-lm-adapt_checkpoint-*.csv"))
            if args.single:
                files = [files[-1]]
            if not files:
                print(f"No files found for seed {seed} and size {size}")
                print("Exiting...")
                exit(1)

            all_scores = {"ppl": [], "p_mean": [], "p_min": [], "aum": []}
            for checkpoint in tqdm(files):
                df = pd.read_csv(checkpoint).fillna("")
                df = df[df["problem"] != "unknown"]
                all_scores["ppl"].append(df["ppl"])
                all_scores["p_mean"].append(-df["p_mean"])
                all_scores["p_min"].append(-df["p_min"])
                all_scores["aum"].append(df["aum"])

            for score_name, scores in all_scores.items():
                scores = [np.array(score).astype(float) for score in scores]
                df[score_name] = np.mean(scores, axis=0)
            df["has_problem"] = ~df["problem"].str.startswith("none")

            dataset_to_problem = {}
            for dataset in df["dataset"].unique():
                df_dataset = df[df["dataset"] == dataset]
                problem = [i for i in df_dataset["problem"].unique() if i != "none"]

                if problem:
                    dataset_to_problem[dataset] = problem[0]
                else:
                    dataset_to_problem[dataset] = "none"

            if args.task_agg == "mean":
                df = df.groupby("dataset").agg({"ppl": "mean", "aum": "mean", "p_mean": "mean", "p_min": "mean", "has_problem": "mean", "problem": "first"})
                df["problem"] = [dataset_to_problem[dataset] for dataset in df.index]
            elif args.task_agg == "median":
                df = df.groupby("dataset").agg({"ppl": "median", "aum": "median", "p_mean": "median", "p_min": "median", "has_problem": "mean", "problem": "first"})
                df["problem"] = [dataset_to_problem[dataset] for dataset in df.index]


            results_df = build_results_df(df)
            problem_to_scores = get_results_rows(results_df)
            for problem, scores in problem_to_scores.items():
                results_rows[problem].append(scores)

        all_lines = [] # should be XX.X±X.X
        for problem in sorted(results_rows.keys()):
            result_strings = []
            mean = np.mean(results_rows[problem], axis=0)
            std = np.std(results_rows[problem], axis=0)
            for m, s in zip(mean, std):
                result_strings.append(f"${m:.1f}_" + "\\textit{" + f"{s:.1f}" + "}$")
            all_lines.append("\t".join([problem] + result_strings))
        size_to_lines[size].extend(all_lines)

    all_lines = []
    for i in range(len(list(size_to_lines.values())[0])):
        line = size_to_lines[sizes[0]][i].split("\t")[:2]
        for size in sizes:
            line.extend(size_to_lines[size][i].split("\t")[2:])
        all_lines.append("\t".join(line))

    results_name = f"results/{args.data}"
    if args.adapted:
        results_name += "_adapted"
    if args.task_agg:
        results_name += f"_{args.task_agg}"
    if args.single:
        results_name += "_single"
    results_name += ".tsv"
    with open(results_name, "w") as f:
        f.write("\n".join(all_lines))



