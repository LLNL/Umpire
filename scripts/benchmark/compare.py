#!/usr/bin/env python3
import os
import argparse
import hatchet as ht

def print_graphs(name, d, depth):
    p=os.path.join(d, "hpctoolkit-database")
    gf = ht.GraphFrame.from_hpctoolkit(p)

    gf.drop_index_levels()

    print("Graph: {} Depth {}".format(name, depth))
    print(gf.tree(metric_column="time (inc)", precision=3, depth=depth, expand_name=True, context_column=None))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare left and right hpctoolkit database with hatched")
    parser.add_argument('lhs')
    parser.add_argument('rhs')

    args = parser.parse_args()
    for x in range(5,6):
        print_graphs("LHS", args.lhs, x)
        print_graphs("RHS", args.rhs, x)
