# ilattice

Generic math on integer lattices (regular grids).

This crate provides types and traits that make it easy to write generic code with integer lattices, i.e. `Z^N`. In
particular, we provide implementations of [`IntegerVector`](crate::vector::IntegerVector) for `glam`'s `IVec2`, `IVec3`,
`UVec2`, and `UVec3` types. There are also some traits that apply to vectors with real number scalars as well, and those are
implemented for `Vec2` and `Vec3A`.

## Example Code
```rust
use ilattice::prelude::*;

// A useful type for looking at bounded subsets of the lattice.
const EXTENT: Extent<IVec3> = Extent::from_min_and_shape(const_ivec3!([-1; 3]), const_ivec3!([5; 3]));

assert!(EXTENT.contains(IVec3::new(-1, 3, 0)));

EXTENT.for_each3(|v| { assert!(EXTENT.contains(v)); });

// Some bitwise logic and shifting code that works for any integer vector.
fn do_vector_math<V: IntegerVector>(v: V, mask: V, shape_log2: V) -> V {
    (v & mask) >> shape_log2
}

let v = do_vector_math(IVec2::new(0xA1, 0xB2), IVec2::new(0xF0, 0xF0), IVec2::new(3, 3));
assert_eq!(v, IVec2::new(20, 22));
```
