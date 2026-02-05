from __future__ import annotations

from pathlib import Path

from .evaluate import run_default_experiments
from .generate_pdf_report import generate_pdf_report


def generate_report(output_path: str = "knn_performance_report.md") -> str:
    df = run_default_experiments()

    lines = []
    lines.append("# k-NN from Scratch: Performance Report")
    lines.append("")
    lines.append("Custom implementation of k-NN with Euclidean and Manhattan distances,")
    lines.append("and an optional k-d tree acceleration. Results obtained via 5-fold")
    lines.append("stratified cross-validation on the local UCI Iris dataset.")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(df.to_markdown(index=False, floatfmt=".4f"))  # type: ignore[arg-type]
    lines.append("")
    lines.append("Columns:")
    lines.append("- **dataset**: dataset name (here always 'iris')")
    lines.append("- **metric**: distance metric used")
    lines.append("- **use_kdtree**: whether k-d tree acceleration was enabled")
    lines.append("- **k_neighbors**: number of neighbors (k)")
    lines.append("- **n_splits**: number of CV folds")
    lines.append("- **mean_accuracy**: average accuracy over folds")
    lines.append("- **std_accuracy**: standard deviation of accuracy")
    lines.append("- **total_runtime_sec**: wall-clock time for all folds")
    lines.append("")

    text = "\n".join(lines)
    Path(output_path).write_text(text, encoding="utf-8")
    return output_path


def main():
    # Generate markdown report
    md_path = generate_report()
    print(f"Markdown report written to {md_path}")
    
    # Generate PDF report (includes performance + coverage)
    pdf_path = generate_pdf_report()
    print(f"PDF report written to {pdf_path}")


if __name__ == "__main__":
    main()

