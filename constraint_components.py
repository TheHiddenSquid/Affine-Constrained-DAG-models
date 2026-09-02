"""Decompose exact affine constraints into their finest independent blocks.

For a system ``A @ omega = b``, invertible row operations may reveal disjoint
sets of variance coordinates even when the supplied rows look coupled.  This
module computes the exact reduced row echelon form of ``[A | b]`` with SymPy,
then uses SciPy's connected-components routine on the RREF fundamental graph.

The resulting blocks form the unique finest decomposition. The routine checks
algebraic consistency, but not positive feasibility of ``omega > 0``.
"""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import List, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ConstraintComponent:
    """An independent reduced constraint block.

    ``columns`` contains the original zero-based variance indices. ``R`` and
    ``c`` define the local exact system ``R @ omega[columns] = c``.
    """

    columns: Tuple[int, ...]
    R: sp.ImmutableMatrix
    c: sp.ImmutableMatrix

    @property
    def rank(self) -> int:
        return self.R.rows


def _exact(value):
    """Convert common Python and NumPy scalars to exact SymPy numbers."""

    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    if isinstance(value, Integral):
        return sp.Integer(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("A and b must contain only finite numbers")
        # Decimal conversion avoids importing the binary expansion of a float.
        return sp.Rational(str(number))
    return sp.Rational(value)


def _coefficient_matrix(A) -> sp.Matrix:
    if hasattr(A, "tolist"):
        A = A.tolist()
    try:
        return sp.Matrix([[_exact(value) for value in row] for row in A])
    except (TypeError, ValueError) as exc:
        raise ValueError("A must be a rectangular matrix") from exc


def _right_hand_side(b) -> sp.Matrix:
    if hasattr(b, "tolist"):
        b = b.tolist()
    values = list(b)
    if values and isinstance(values[0], (list, tuple)):
        if any(len(value) != 1 for value in values):
            raise ValueError("b must be a vector or a one-column matrix")
        values = [value[0] for value in values]
    return sp.Matrix([_exact(value) for value in values])


def decompose_affine_constraints(
    A: Sequence[Sequence[object]], b: Sequence[object]
) -> List[ConstraintComponent]:
    """Return the unique finest independent components of ``A @ omega = b``.

    The input is treated as exact. Integer and rational entries remain exact;
    floating-point entries are converted from their displayed decimal strings.
    Therefore a tiny nonzero coefficient remains nonzero and couples its blocks.

    Columns absent from every returned component are unconstrained.  Use
    :func:`unconstrained_columns` to list them.
    """

    A_matrix = _coefficient_matrix(A)
    b_vector = _right_hand_side(b)
    if A_matrix.rows != b_vector.rows:
        raise ValueError("A and b must have the same number of rows")
    if A_matrix.rows == 0:
        return []

    number_of_variables = A_matrix.cols
    augmented_rref, pivots = A_matrix.row_join(b_vector).rref()

    # A pivot in the augmented column is exactly an inconsistent row 0 = 1.
    if number_of_variables in pivots:
        raise ValueError("The affine system A omega = b is inconsistent")

    rank = len(pivots)
    if rank == 0:
        return []

    # SymPy places the nonzero RREF rows first, one for each pivot.
    R = augmented_rref[:rank, :number_of_variables]
    c = augmented_rref[:rank, number_of_variables]

    # In the fundamental graph, every nonzero R[k, j] connects variable j to
    # the pivot variable of row k.  Its connected components are precisely the
    # column-matroid components and hence the finest possible disjoint blocks.
    nonzeros_by_row = defaultdict(list)
    for row, column in R.todok():
        nonzeros_by_row[row].append(column)

    active_columns = sorted(
        {column for columns in nonzeros_by_row.values() for column in columns}
    )
    position = {column: index for index, column in enumerate(active_columns)}

    graph_rows = []
    graph_columns = []
    for row, pivot in enumerate(pivots):
        pivot_position = position[pivot]
        for column in nonzeros_by_row[row]:
            column_position = position[column]
            if column_position != pivot_position:
                graph_rows.extend((pivot_position, column_position))
                graph_columns.extend((column_position, pivot_position))

    adjacency = coo_matrix(
        (
            np.ones(len(graph_rows), dtype=bool),
            (graph_rows, graph_columns),
        ),
        shape=(len(active_columns), len(active_columns)),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)

    grouped_columns = defaultdict(list)
    for column, label in zip(active_columns, labels):
        grouped_columns[int(label)].append(column)
    column_components = sorted(
        (tuple(columns) for columns in grouped_columns.values()),
        key=lambda columns: columns[0],
    )

    components = []
    for columns in column_components:
        column_set = set(columns)
        rows = [
            row
            for row, pivot in enumerate(pivots)
            if pivot in column_set
        ]
        components.append(
            ConstraintComponent(
                columns=columns,
                R=R.extract(rows, columns).as_immutable(),
                c=sp.ImmutableMatrix([c[row] for row in rows]),
            )
        )
    return components


def unconstrained_columns(
    number_of_variables: int, components: Sequence[ConstraintComponent]
) -> Tuple[int, ...]:
    """Return the original variable indices that do not occur in ``A``."""

    active = {column for component in components for column in component.columns}
    return tuple(column for column in range(number_of_variables) if column not in active)


def main():
    A = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, -1, 0],
                [0, 0, 0, 0, 0, 1, -1, 0]])
    b = np.array([2, 6, 0, 0])
    p = A.shape[1]

    components = decompose_affine_constraints(A, b)

    for component in components:
        print(component.columns, component.R, component.c, component.rank)

    print(unconstrained_columns(p, components))


if __name__ == "__main__":
    main()
