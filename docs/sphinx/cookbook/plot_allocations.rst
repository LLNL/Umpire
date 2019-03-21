.. _plot_allocations:

======================
Visualizing Allocators
======================

The python script `plot_allocations.py` is included with Umpire to
plot allocations. This script uses series of three arguments: an
output file with allocation records, a color, and an alpha
(transparency) value `0.0-1.0`. Although these could be used to plot
records from a single allocator, 3 arguments, it can also be used to
overlay multiple allocators, by passing 3n arguments after the script
name. In this cookbook we use this feature to visualize a pooled allocator.

The cookbook generates two files, `allocator.log` and
`pooled_allocator.log`, that contain the allocation records from the
underlying allocator and the pool. These can then be plotted using a
command similar to the following:

```
tools/plot_allocations allocator.log gray 0.2 pooled_allocator.log purple 0.8
```

That script uses Python and Matplotlib to generate the following image

.. image: plot_allocations_example.png

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_plot_allocations.cpp
