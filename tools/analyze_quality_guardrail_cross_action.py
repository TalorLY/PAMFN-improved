"""Aggregate the frozen PCS/Ball guardrail-v2 screening decisions."""

import argparse
import json
import os


ACTIONS = ("PCS", "Ball")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcs_report")
    parser.add_argument("--ball_report")
    parser.add_argument("--tes_report")
    parser.add_argument("--output")
    parser.add_argument("--base_no_regret", type=float, default=-0.10)
    parser.add_argument("--signal_threshold", type=float, default=0.50)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and not all((
            args.pcs_report, args.ball_report,
            args.tes_report, args.output)):
        parser.error(
            "PCS, Ball, TES reports and --output are required"
        )
    return args


def load_report(path, expected_action):
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    action = report.get("action", "TES")
    if action != expected_action:
        raise ValueError(
            f"{path} belongs to {action}, expected {expected_action}"
        )
    for split in ("train", "cal"):
        if not {"off", "prior_only", "guardrail"}.issubset(
                report["metrics"][split]):
            raise ValueError(f"{path} is missing required variants")
    return report


def summarize(report):
    metrics = report["metrics"]["cal"]
    return {
        "decision": report["decision"]["status"],
        "base_coef": metrics["off"]["coef"],
        "prior_only_coef": metrics["prior_only"]["coef"],
        "guardrail_coef": metrics["guardrail"]["coef"],
        "guardrail_vs_base_coef": (
            metrics["guardrail"]["coef"] - metrics["off"]["coef"]
        ),
        "guardrail_vs_prior_only_coef": (
            metrics["guardrail"]["coef"]
            - metrics["prior_only"]["coef"]
        ),
        "motion_prior": report["task_priors"]["motion"],
        "skeleton_prior": report["task_priors"]["skeleton"],
    }


def aggregate(tes, pcs, ball, base_no_regret, signal_threshold):
    summaries = {
        "TES": summarize(tes),
        "PCS": summarize(pcs),
        "Ball": summarize(ball),
    }
    screened = [summaries[action] for action in ACTIONS]
    individual_pass = all(
        value["decision"] == "GO_NEXT_TASKS" for value in screened
    )
    base_no_regret_pass = all(
        value["guardrail_vs_base_coef"] >= base_no_regret
        for value in screened
    )
    cross_action_signal_pass = max(
        value["guardrail_vs_base_coef"] for value in screened
    ) >= signal_threshold
    tes_prior = (
        summaries["TES"]["motion_prior"],
        summaries["TES"]["skeleton_prior"],
    )
    prior_diversity = {
        action: max(
            abs(summaries[action]["motion_prior"] - tes_prior[0]),
            abs(summaries[action]["skeleton_prior"] - tes_prior[1]),
        )
        for action in ACTIONS
    }
    passed = (
        individual_pass
        and base_no_regret_pass
        and cross_action_signal_pass
    )
    return {
        "schema_version": 1,
        "protocol": (
            "frozen guardrail-v2 PCS/Ball seed-0 screen; Ball uses "
            "100 screening epochs, not its final 500-epoch schedule"
        ),
        "thresholds": {
            "per_action_guardrail_vs_base_min_coef": base_no_regret,
            "at_least_one_new_action_signal_coef": signal_threshold,
        },
        "actions": summaries,
        "task_prior_max_absolute_difference_from_TES": prior_diversity,
        "decision": {
            "individual_action_checks_pass": individual_pass,
            "base_no_regret_pass": base_no_regret_pass,
            "cross_action_signal_pass": cross_action_signal_pass,
            "status": (
                "GO_FULL_EXPERIMENT" if passed
                else "STOP_AND_REASSESS"
            ),
            "note": (
                "GO authorizes the six-action multi-seed experiment with "
                "the method frozen; it is not itself a final result."
            ),
        },
    }


def fake_report(action, base, guardrail, status="GO_NEXT_TASKS"):
    return {
        "action": action,
        "metrics": {
            split: {
                "off": {"coef": base},
                "prior_only": {"coef": guardrail - 0.05},
                "guardrail": {"coef": guardrail},
            }
            for split in ("train", "cal")
        },
        "task_priors": {"motion": 0.0, "skeleton": 0.8},
        "decision": {"status": status},
    }


def self_test():
    tes = fake_report("TES", 55.0, 56.0)
    pcs = fake_report("PCS", 60.0, 60.6)
    ball = fake_report("Ball", 70.0, 70.0)
    report = aggregate(tes, pcs, ball, -0.10, 0.50)
    assert report["decision"]["status"] == "GO_FULL_EXPERIMENT"
    ball["metrics"]["cal"]["guardrail"]["coef"] = 69.8
    report = aggregate(tes, pcs, ball, -0.10, 0.50)
    assert report["decision"]["status"] == "STOP_AND_REASSESS"
    print("quality guardrail cross-action analyzer self-test: PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    reports = {
        "TES": load_report(args.tes_report, "TES"),
        "PCS": load_report(args.pcs_report, "PCS"),
        "Ball": load_report(args.ball_report, "Ball"),
    }
    report = aggregate(
        reports["TES"], reports["PCS"], reports["Ball"],
        args.base_no_regret, args.signal_threshold,
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
