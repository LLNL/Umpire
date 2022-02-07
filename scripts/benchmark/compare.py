#!/usr/bin/env python3
import os
import argparse
import hatchet as ht

def print_graphs(name, d):
    p=os.path.join(d, "hpctoolkit-database")
    gf_event = ht.GraphFrame.from_hpctoolkit(p)

    gf_event.drop_index_levels()

    print("Graph: {}".format(name))
    # for x in range(5, 11):
    for x in range(5, 6):
        print(gf_event.tree(metric_column="time (inc)", precision=3, depth=x, expand_name=True, context_column=None))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare left and right hpctoolkit database with hatched")
    parser.add_argument('lhs')
    parser.add_argument('rhs')

    args = parser.parse_args()
    print_graphs("LHS", args.lhs)
    print_graphs("RHS", args.rhs)
