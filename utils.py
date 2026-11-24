"""Small utility functions used by the Streamlit application."""
def suitability_message(label):
    if str(label).lower() in {"suitable", "yes", "high", "1"}:
        return "The model predicts this soil sample is suitable under the dataset's learning labels. Consider confirming with a soil test and local agronomy advice."
    return "The model predicts this soil sample may need improvement under the dataset's learning labels. Consider soil testing and suitable soil-management practices."

def format_metrics(metrics):
    return {name: round(float(value), 3) for name, value in metrics.items()}
