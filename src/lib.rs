//! An iterator-based implementation of the Amantides and Woo algorithm for casting a ray through pixels or voxels.

pub use ilattice;

use ilattice::prelude::{FloatVector, IntegerVector, SignedVector, Vector2, Vector3};

// Note: we need newtypes for 2D and 3D because rust doesn't have specialization to figure out if it should use the `Vector2` or
// `Vector3` trait impls

/// Visits every pixel intersecting the given 2D ray.
pub struct GridRayIter2<Vi, Vf> {
    inner: GridRayIter<Vi, Vf>,
}

impl<Vi, Vf> GridRayIter2<Vi, Vf>
where
    Vi: IntegerVector<Float = Vf>,
    Vf: FloatVector<Int = Vi> + SignedVector,
{
    /// Initialize the traversal, beginning at the `start` position and moving along the `direction` vector.
    pub fn new(start: Vf, direction: Vf) -> Self {
        Self {
            inner: GridRayIter::new(start, direction),
        }
    }
}

/// Visits every voxel intersecting the given 3D ray.
pub struct GridRayIter3<Vi, Vf> {
    inner: GridRayIter<Vi, Vf>,
}

impl<Vi, Vf> GridRayIter3<Vi, Vf>
where
    Vi: IntegerVector<Float = Vf>,
    Vf: FloatVector<Int = Vi> + SignedVector,
{
    /// Initialize the traversal, beginning at the `start` position and moving along the `direction` vector.
    pub fn new(start: Vf, direction: Vf) -> Self {
        Self {
            inner: GridRayIter::new(start, direction),
        }
    }
}

/// The generic type underlying `GridRayIter2` and `GridRayIter3`.
pub struct GridRayIter<Vi, Vf> {
    // The current pixel/voxel position.
    current_grid_point: Vi,
    // Either -1 or +1 in each axis. The direction we step along each axis.
    step: Vi,
    // The amount of time it takes to move 1 unit along each axis.
    t_delta: Vf,
    // The next time when each axis will cross a pixel boundary.
    t_max: Vf,
}

impl<Vi, Vf> GridRayIter<Vi, Vf>
where
    Vi: IntegerVector<Float = Vf>,
    Vf: FloatVector<Int = Vi> + SignedVector,
{
    /// Initialize the traversal, beginning at the `start` position and moving along the `direction` vector.
    #[inline]
    pub fn new(start: Vf, direction: Vf) -> Self {
        let current_grid_point: Vi = start.cast();
        let vel_signs = direction.signum();
        let step: Vi = vel_signs.cast();
        let t_delta = vel_signs / direction;

        // For each axis, calculate the time delta we need to reach a pixel boundary on that axis. For a positive direction,
        // this is just the next pixel, but for negative, it's the current pixel (hence the LUB with zero).
        let next_bounds: Vf = (current_grid_point + step.least_upper_bound(Vi::ZERO)).cast();
        let delta_to_next_bounds = next_bounds - start;
        let t_max = delta_to_next_bounds / direction;

        Self {
            current_grid_point,
            step,
            t_delta,
            t_max,
        }
    }
}

impl<Vi, Vf> GridRayIter<Vi, Vf>
where
    Vi: Vector2,
    Vf: Vector2,
{
    /// Move to the next closest pixel along the ray.
    #[inline]
    pub fn step2(&mut self) {
        if self.t_max.x() < self.t_max.y() {
            *self.current_grid_point.x_mut() += self.step.x();
            *self.t_max.x_mut() += self.t_delta.x();
        } else {
            *self.current_grid_point.y_mut() += self.step.y();
            *self.t_max.y_mut() += self.t_delta.y();
        }
    }

    #[inline]
    pub fn current_pixel(&self) -> Vi {
        self.current_grid_point
    }
}

impl<Vi, Vf> GridRayIter<Vi, Vf>
where
    Vi: Vector3,
    Vf: Vector3,
{
    /// Move to the next closest voxel along the ray.
    #[inline]
    pub fn step3(&mut self) {
        if self.t_max.x() < self.t_max.y() {
            if self.t_max.x() < self.t_max.z() {
                *self.current_grid_point.x_mut() += self.step.x();
                *self.t_max.x_mut() += self.t_delta.x();
            } else {
                *self.current_grid_point.z_mut() += self.step.z();
                *self.t_max.z_mut() += self.t_delta.z();
            }
        } else if self.t_max.y() < self.t_max.z() {
            *self.current_grid_point.y_mut() += self.step.y();
            *self.t_max.y_mut() += self.t_delta.y();
        } else {
            *self.current_grid_point.z_mut() += self.step.z();
            *self.t_max.z_mut() += self.t_delta.z();
        }
    }

    #[inline]
    pub fn current_voxel(&self) -> Vi {
        self.current_grid_point
    }
}

impl<Vi, Vf> Iterator for GridRayIter2<Vi, Vf>
where
    Vi: Vector2,
    Vf: Vector2,
{
    type Item = Vi;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.step2();
        Some(self.inner.current_pixel())
    }
}

impl<Vi, Vf> Iterator for GridRayIter3<Vi, Vf>
where
    Vi: Vector3,
    Vf: Vector3,
{
    type Item = Vi;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.step3();
        Some(self.inner.current_voxel())
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod tests {
    use ilattice::glam::{const_ivec3, const_vec3a};

    use super::*;

    #[test]
    fn test_move_along_x_axis() {
        let mut traversal =
            GridRayIter::new(const_vec3a!([0.5, 0.5, 0.5]), const_vec3a!([1.0, 0.0, 0.0]));

        println!("STEP = {:?}", traversal.t_max);

        let mut voxels = Vec::new();
        for _ in 0..5 {
            voxels.push(traversal.current_voxel());
            traversal.step3();
        }

        assert_eq!(
            voxels,
            vec![
                const_ivec3!([0, 0, 0]),
                const_ivec3!([1, 0, 0]),
                const_ivec3!([2, 0, 0]),
                const_ivec3!([3, 0, 0]),
                const_ivec3!([4, 0, 0])
            ]
        )
    }

    #[test]
    fn test_move_along_all_axes_some_negative() {
        let mut traversal = GridRayIter::new(
            const_vec3a!([0.5, 0.5, 0.5]),
            const_vec3a!([1.0, -2.0, 3.0]),
        );

        let mut voxels = Vec::new();
        for _ in 0..10 {
            voxels.push(traversal.current_voxel());
            traversal.step3();
        }

        assert_eq!(
            voxels,
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
        )
    }
}
