import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    args, v = parser.parse_known_args()

    print(args)
    print(v)
# print(parser)
