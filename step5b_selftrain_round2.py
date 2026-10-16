"""
step5b_selftrain_round2.py — Self-Training Round 2
────────────────────────────────────────────────────────────────────

"""

from step5_selftrain_common import run_selftrain_round, build_argparser


def main():
    args = build_argparser(round_num=2).parse_args()
    run_selftrain_round(round_num=2, pipelines=args.pipeline)


if __name__ == "__main__":
    main()
