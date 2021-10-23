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
        const_ivec3!([0, 0, 0]),
        const_ivec3!([0, 0, 1]),
        const_ivec3!([0, -1, 1]),
        const_ivec3!([0, -1, 2]),
        const_ivec3!([1, -1, 2]),
        const_ivec3!([1, -2, 2]),
        const_ivec3!([1, -2, 3]),
        const_ivec3!([1, -2, 4]),
        const_ivec3!([1, -3, 4]),
        const_ivec3!([2, -3, 4]),
    ]
);
```
