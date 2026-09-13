Housing Utilities
=================

Custom Transformers
-------------------

HousingLogTransformer
~~~~~~~~~~~~~~~~~~~~~

The ``HousingLogTransformer`` applies a reversible signed logarithmic
transformation to numeric features. It is useful for reducing the influence
of large-magnitude values while preserving the sign of the original values.

The transformation is:

.. math::

   f(x) = \operatorname{sign}(x)\log(1 + |x|)

The original values can be recovered with ``inverse_transform``.

