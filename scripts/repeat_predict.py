import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    for i in range(args.n):
        print(f"\n=== RUN {i+1}/{args.n} ===")
        cmd = ["python", "scripts/predict.py", "--config", args.config] + args.extra
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
