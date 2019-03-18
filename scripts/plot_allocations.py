#!/usr/bin/env python3

import matplotlib.pyplot as plt

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

def find_scale(min_byte, max_byte):
    scales = ('Bytes', 'kB', 'MB', 'GB')

    # Loop until x-axis will have numbers < 10000
    expo = 0
    while (max_byte - min_byte) / (1024 ** expo) > 1e4:
        expo += 1

    # Divide by this
    denom = 1024 ** expo

    return denom, scales[expo]

def plot_ranges(*args):
    '''Plots memory ranges for each dict in args. Each dict must have the
    fields ranges, color, and alpha.'''

    from matplotlib.collections import PolyCollection

    f = plt.figure(figsize=(12,1), tight_layout=True)
    axes = f.gca()

    # Find the correct scale to use
    lowest = min((record_set['ranges'][0][0] for record_set in args))
    highest = max((record_set['ranges'][-1][1] for record_set in args))

    denom, scale = find_scale(lowest, highest)

    # Add the rectangles from each record set in args
    for record_set in args:
        verts = [ ((a[0] / denom, 0), (a[1] / denom, 0), (a[1] / denom, 1), (a[0] / denom, 1)) for a in record_set['ranges'] ]
        rects = PolyCollection(verts, closed=True, color=record_set['color'], alpha=record_set['alpha'])
        axes.add_collection(rects)
    axes.autoscale_view()
    axes.set_yticks(ticks=())
    axes.set_xlabel('Offset relative to lowest address ({:s})'.format(scale))

    return f

if __name__ == '__main__':
    from sys import argv

    usage = 'USAGE: plot_allocations.py [ LOGFILE color alpha ] ...'

    all_records = []
    index = 1
    while index < len(argv):
        allocs = import_from_file(argv[index+0])

        if len(argv) < index+3:
            raise ValueError(usage)

        start = allocs[0][0]
        coalesced_allocs = coalesce_records([ (a[0]-start, a[1]-start) for a in allocs ])

        all_records.append({'ranges':coalesced_allocs, 'color':argv[index+1], 'alpha':argv[index+2]})

        index += 3

    if len(all_records) > 0:
        f1 = plot_ranges(*all_records)
    else:
        raise ValueError('No records found. {:s}'.format(usage));

    plt.show()

# These functions are not used
def invert_segments(slist):
    return [(slist[i][1], slist[i+1][0]) for i in range(len(slist)-1)]

def plot_histogram(nbins, slist, vmin=None, vmax=None):
    '''Plot a histogram of fragmentation. This is not currently used, but
    included for possible future use.

    Example usage:
    % gaps = invert_segments(allocs)
    % f2 = plot_histogram(10, gaps, vmin=allocs[0][0], vmax=allocs[-1][1])
    '''
    def overlap(s1, s2):
        '''Return the overlap of s1 and s2'''
        return max(0, min(s1[1], s2[1]) - max(s1[0], s2[0]))

    f = plt.figure(figsize=(6,3))
    axes = f.gca()

    # Find the correct scale to use
    if vmin is None: vmin = slist[0][0]
    if vmax is None: vmax = slist[-1][1]

    denom, scale = find_scale(vmin, vmax)

    # Create bins
    overlaps = [0] * nbins
    ds = (vmax - vmin) / (nbins + 1)
    bins = [(vmin + ds * bi, vmin + ds * (bi+1)) for bi in range(nbins)]

    # Determine how much of each address range falls inside each bin
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

    # Data to plot
    yaxis = [o/ds * 100 for o in overlaps]
    xaxis = [(b[0] + ds/2)/denom for b in bins]

    axes.bar(xaxis, yaxis, ds/1.2/denom)
    axes.autoscale_view()
    axes.set_xlabel('Address bin')
    axes.set_ylabel('Relative fragmentation (%)')

    return f
