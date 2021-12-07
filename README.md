# grid-ray

An iterator-based implementation of the Amantides and Woo algorithm for casting a ray through pixels or voxels.

## Code Example

```rust
use grid_ray::GridRayIter3;
use grid_ray::ilattice::glam::{const_ivec3, const_vec3a};

let start = const_vec3a!([0.5, 0.5, 0.5]);
let direction = const_vec3a!([1.0, -2.0, 3.0]);
let mut traversal = GridRayIter3::new(start, direction);

assert_eq!(
    traversal.take(10).collect::<Vec<_>>(),
    vec![
        (0.0, const_ivec3!([0, 0, 0])),
        (0.16666667, const_ivec3!([0, 0, 1])),
        (0.25, const_ivec3!([0, -1, 1])),
        (0.5, const_ivec3!([0, -1, 2])),
        (0.5, const_ivec3!([1, -1, 2])),
        (0.75, const_ivec3!([1, -2, 2])),
        (0.8333334, const_ivec3!([1, -2, 3])),
        (1.1666667, const_ivec3!([1, -2, 4])),
        (1.25, const_ivec3!([1, -3, 4])),
        (1.5, const_ivec3!([2, -3, 4])),
    ]
);
```
