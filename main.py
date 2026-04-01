from src.explain import run_shap_analysis
from src.train import train_and_select_model


def main() -> None:
    artifacts = train_and_select_model()
    run_shap_analysis(artifacts)


if __name__ == "__main__":
    main()
