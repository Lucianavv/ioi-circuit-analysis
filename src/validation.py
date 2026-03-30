import numpy as np
from typing import Set, Tuple, Dict
from src.acdc import compute_circuit_overlap


# Known IOI circuit heads from Wang et al. 2022
KNOWN_IOI_HEADS = {
    "name_movers":          [(9, 9), (9, 6), (10, 0)],
    "backup_name_movers":   [(10, 7), (11, 10)],
    "s_inhibition":         [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction":            [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token":      [(0, 1), (0, 10), (3, 0)],
    "previous_token":       [(2, 2), (4, 11)]
}

ALL_KNOWN_HEADS = {
    (l, h)
    for heads in KNOWN_IOI_HEADS.values()
    for (l, h) in heads
}


def evaluate_circuit_recovery(
    circuit: Set[Tuple[int, int]],
    algorithm_name: str
) -> Dict:

    results = {}

    for head_type, heads in KNOWN_IOI_HEADS.items():
        known = set(heads)
        found = known & circuit
        recovery = len(found) / len(known) if known else 0.0
        results[head_type] = {
            "known": sorted(known),
            "found": sorted(found),
            "missed": sorted(known - circuit),
            "recovery": recovery
        }

    overall = len(ALL_KNOWN_HEADS & circuit) / len(ALL_KNOWN_HEADS)
    results["overall_recovery"] = overall

    print(f"\n{'='*60}")
    print(f"Circuit Recovery — {algorithm_name}")
    print(f"{'='*60}")
    print(f"{'Head Type':<25} {'Recovery':<12} {'Found':<8} {'Missed'}")
    print(f"{'-'*60}")

    for head_type, r in results.items():
        if head_type == "overall_recovery":
            continue
        print(f"{head_type:<25} {r['recovery']*100:<12.1f}% "
              f"{r['found']} | missed: {r['missed']}")

    print(f"\nOverall recovery: {overall*100:.1f}% "
          f"({len(ALL_KNOWN_HEADS & circuit)}/{len(ALL_KNOWN_HEADS)} "
          f"known heads found)")

    return results


def compare_methods(
    acdc_circuit: Set[Tuple[int, int]],
    ap_circuit: Set[Tuple[int, int]],
    acdc_metrics: Dict,
    ap_metrics: Dict
) -> None:

    overlap = compute_circuit_overlap(acdc_circuit, ap_circuit)

    print(f"\n{'='*60}")
    print(f"Method Comparison — ACDC vs Attribution Patching")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'ACDC':<15} {'Attr. Patching'}")
    print(f"{'-'*60}")
    print(f"{'Faithfulness':<30} "
          f"{acdc_metrics['faithfulness']:<15.3f} "
          f"{ap_metrics['faithfulness']:.3f}")
    print(f"{'Sparsity':<30} "
          f"{acdc_metrics['sparsity']:<15.3f} "
          f"{ap_metrics['sparsity']:.3f}")
    print(f"{'Circuit size (heads)':<30} "
          f"{acdc_metrics['circuit_size']:<15} "
          f"{ap_metrics['circuit_size']}")
    print(f"{'Forward passes':<30} "
          f"{acdc_metrics['forward_passes']:<15} "
          f"{ap_metrics['forward_passes']}")
    print(f"{'Circuit overlap (Jaccard)':<30} {overlap:.3f}")

    # Heads in ACDC but not AP
    only_acdc = acdc_circuit - ap_circuit
    only_ap = ap_circuit - acdc_circuit
    both = acdc_circuit & ap_circuit

    print(f"\nIn both circuits:     {sorted(both)}")
    print(f"Only in ACDC:         {sorted(only_acdc)}")
    print(f"Only in Attr. Patch:  {sorted(only_ap)}")


def run_full_validation(
    acdc_circuit: Set[Tuple[int, int]],
    ap_circuit: Set[Tuple[int, int]],
    acdc_metrics: Dict,
    ap_metrics: Dict
) -> None:

    print("\n" + "="*60)
    print("FULL VALIDATION REPORT")
    print("="*60)

    # Recovery analysis
    acdc_recovery = evaluate_circuit_recovery(
        acdc_circuit, "ACDC")
    ap_recovery = evaluate_circuit_recovery(
        ap_circuit, "Attribution Patching")

    # Method comparison
    compare_methods(
        acdc_circuit, ap_circuit,
        acdc_metrics, ap_metrics)

    # Pass/fail assessment
    print(f"\n{'='*60}")
    print("PASS/FAIL ASSESSMENT")
    print(f"{'='*60}")

    acdc_pass = (
        acdc_metrics['faithfulness'] > 0.5 and
        acdc_recovery['overall_recovery'] > 0.5
    )
    ap_pass = (
        ap_metrics['faithfulness'] > 0.5 and
        ap_recovery['overall_recovery'] > 0.5
    )

    print(f"ACDC implementation:              "
          f"{'PASS' if acdc_pass else 'FAIL'}")
    print(f"Attribution Patching implementation: "
          f"{'PASS' if ap_pass else 'FAIL'}")

    if not acdc_pass or not ap_pass:
        print("\nNote: Consider adjusting threshold tau and re-running.")
        print("Lower tau = larger circuit = higher faithfulness.")
        print("Higher tau = smaller circuit = higher sparsity.")
