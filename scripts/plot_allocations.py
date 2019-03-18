#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from re import compile

a_pat = compile(r'\s*(?P<bytes>\d+)\s+\[\s+(?P<start>\w+)\s+--\s+(?P<end>\w+)\s*\]')

def import_from_file(fname):
    with open(fname, mode='r') as f:
        return [(int(m.group('start'), 16), int(m.group('end'), 16)) \
                for m in (a_pat.match(l) for l in f) if m]

def coalesce_records(slist):
    smaller = [list(slist[0])]
    for s in slist[1:]:
        if s[0] == smaller[-1][1] + 1:
            smaller[-1][1] = s[1]
        else:
            smaller.append(list(s))
    return smaller

def invert_segments(slist):
    return [(slist[i][1], slist[i+1][0]) for i in range(len(slist)-1)]

def plot_ranges(axes, *args):
    for record_set in args:
        verts = [ ((a[0], 0), (a[1], 0), (a[1], 1), (a[0], 1)) for a in record_set['ranges'] ]
        rects = PolyCollection(verts, closed=True, color=record_set['color'], alpha=record_set['alpha'])
        axes.add_collection(rects)
        axes.autoscale_view()
    axes.set_yticks(ticks=())
    axes.set_xlabel('Address offset (bytes)')

def plot_histogram(axes, nbins, slist, vmin=None, vmax=None):
    def overlap(s1, s2):
        '''Return the overlap of s1 and s2'''
        return max(0, min(s1[1], s2[1]) - max(s1[0], s2[0]))

    if vmin is None: vmin = slist[0][0]
    if vmax is None: vmax = slist[-1][1]

    overlaps = [0] * nbins
    ds = (vmax - vmin) / (nbins + 1)
    bins = [(vmin + ds * bi, vmin + ds * (bi+1)) for bi in range(nbins)]

    ri = 0
    for bi in range(nbins):
        curr_bin = bins[bi]
        while True:
            o = overlap(curr_bin, slist[ri])
            if o > 0:
                overlaps[bi] += o
                ri += 1
            elif slist[ri][1] - slist[ri][0] == 0:
                ri += 1
            else:
                break

    yaxis = [o/ds * 100 for o in overlaps]
    xaxis = [b[0] + ds/2 for b in bins]
    print(yaxis, xaxis)

    axes.bar(xaxis, yaxis, ds/1.2)
    axes.autoscale_view()


if __name__ == '__main__':
    from sys import argv
    allocs = import_from_file(argv[1])

    start = allocs[0][0]
    coalesced_allocs = coalesce_records([ (a[0]-start, a[1]-start) for a in allocs ])

    f1 = plt.figure(figsize=(12,1), tight_layout=True)
    a1 = f1.gca()

    all_records = ({'ranges':coalesced_allocs, 'color':'gray', 'alpha':1.0},)
    plot_ranges(a1, *all_records)
    # plt.savefig('address_usage.pdf', bbox_inches='tight')

    # f2 = plt.figure(figsize=(6,3))
    # a2 = f2.gca()

    # gaps = invert_segments(fewer_allocs)
    # plot_histogram(a2, 10, gaps, vmin=fewer_allocs[0][0], vmax=fewer_allocs[-1][1])

    # a2.set_xlabel('Address ranges')
    # a2.set_ylabel('Relative fragmentation (%)')
    # # f2.savefig('relative_fragmentation.pdf', bbox_inches='tight')

    plt.show()
