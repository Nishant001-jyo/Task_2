"""Run the full Week 2 pipeline: regression + classification."""
from src import regression, classification


def main():
    reg = regression.run()
    clf = classification.run()
    print("=" * 60)
    print("Done. Tables and plots saved in ./results/")
    print("Best regressor  :", reg["R2"].idxmax())
    print("Best classifier :", clf["F1"].idxmax())


if __name__ == "__main__":
    main()
