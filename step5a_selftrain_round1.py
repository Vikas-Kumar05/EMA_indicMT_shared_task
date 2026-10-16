"""
step5a_selftrain_round1.py — Self-Training Round 1
────────────────────────────────────────────────────────────────────

"""

from step5_selftrain_common import run_selftrain_round, build_argparser


def main():
    args = build_argparser(round_num=1).parse_args()
    run_selftrain_round(round_num=1, pipelines=args.pipeline)


if __name__ == "__main__":
    main()
