from src.docvisrag.config import ProjectConfig


def main() -> None:
    cfg = ProjectConfig()
    print("DocVisRAG project scaffold is ready.")
    print(f"Default model_id: {cfg.model_id}")
    print("Run `python scripts/check_env.py` for environment diagnostics.")


if __name__ == "__main__":
    main()
