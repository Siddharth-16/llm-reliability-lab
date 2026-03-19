from src.eval.runner import run_evaluation
from src.eval.report.save_results import save_report

def main():
    report = run_evaluation("datasets/sample_corpus/questions.jsonl")

    path = save_report(report)

    print("\nEvaluation Summary:\n")
    print(report["summary"])

    print(f"\nReport saved to: {path}")

if __name__ == "__main__":
    main()