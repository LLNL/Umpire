.. _strategy_name:

=============================
Getting the Strategy Name
=============================

Since every Allocator is represented by the same type after it's been
created, it can be difficult to determine exactly what kind of strategy the
allocator is using. The name of the strategy can be accessed using the
`Allocator::getStrategyName()` method:

.. literalinclude:: ../../../examples/cookbook/recipe_strategy_name.cpp
