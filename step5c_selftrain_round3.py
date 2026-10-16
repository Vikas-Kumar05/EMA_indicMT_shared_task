"""
step5c_selftrain_round3.py — Self-Training Round 3 (final round)
────────────────────────────────────────────────────────────────────

"""

from step5_selftrain_common import run_selftrain_round, build_argparser


def main():
    args = build_argparser(round_num=3).parse_args()
    run_selftrain_round(round_num=3, pipelines=args.pipeline)


if __name__ == "__main__":
    main()
