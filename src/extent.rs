use crate::vector::*;

use core::ops::{Add, Mul, Shl, Shr, Sub};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An N-dimensional extent. This is mathematically the Cartesian product of a half-closed interval `[a, b)` in each dimension.
/// You can also just think of it as an axis-aligned box with some shape and a minimum point.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Extent<V> {
    /// The least point contained in the extent.
    pub minimum: V,
    /// The length of each dimension.
    pub shape: V,
}

// A few of these traits could be derived. But it seems that derive will not help the compiler infer trait bounds as well.
impl<V> Clone for Extent<V>
where
    V: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            minimum: self.minimum.clone(),
            shape: self.shape.clone(),
        }
    }
}
impl<V> Copy for Extent<V> where V: Copy {}
impl<V> PartialEq for Extent<V>
where
    V: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.minimum.eq(&other.minimum) && self.shape.eq(&other.shape)
    }
}
impl<V> Eq for Extent<V> where V: Eq {}

impl<V> Extent<V> {
    /// The default representation of an extent as the minimum point and shape.
    #[inline]
    pub const fn from_min_and_shape(minimum: V, shape: V) -> Self {
        Self { minimum, shape }
    }

    #[inline]
    pub fn map_components<T>(&self, f: impl Fn(&V) -> T) -> Extent<T> {
        Extent {
            minimum: f(&self.minimum),
            shape: f(&self.shape),
        }
    }
}

impl<V> Extent<V>
where
    V: Vector,
{
    /// An alternative representation of an extent as the minimum point and least upper bound.
    #[inline]
    pub fn from_min_and_lub(minimum: V, least_upper_bound: V) -> Self {
        let minimum = minimum;
        // We want to avoid negative shape components.
        let shape = (least_upper_bound - minimum).least_upper_bound(V::ZERO);

        Self { minimum, shape }
    }

    /// Translate the extent such that it has `new_min` as it's new minimum.
    #[inline]
    pub fn with_minimum(&self, new_min: V) -> Self {
        Self::from_min_and_shape(new_min, self.shape)
    }

    /// Resize the extent such that it has `new_shape` as it's new shape.
    #[inline]
    pub fn with_shape(&self, new_shape: V) -> Self {
        Self::from_min_and_shape(self.minimum, new_shape)
    }

    /// The least point `p` for which all points `q` in the extent satisfy `q < p`.
    #[inline]
    pub fn least_upper_bound(&self) -> V {
        self.minimum + self.shape
    }

    /// The number of points in the extent.
    #[inline]
    pub fn volume(&self) -> V::Scalar {
        self.shape.fold(V::Scalar::ONE, |c, out| c * out)
    }

    /// Returns `true` iff the point `p` is contained in this extent.
    #[inline]
    pub fn contains(&self, p: V) -> bool {
        let lub = self.least_upper_bound();

        self.minimum.with_lattice_ord() <= p.with_lattice_ord()
            && p.with_lattice_ord() < lub.with_lattice_ord()
    }

    /// Returns a new extent that's been padded on all borders by `pad_amount`.
    #[inline]
    pub fn padded(&self, pad_amount: V::Scalar) -> Self {
        Self::from_min_and_shape(
            self.minimum - V::splat(pad_amount),
            self.shape + V::splat(pad_amount + pad_amount),
        )
    }

    /// Returns `Some(self)` iff this extent has a positive shape, otherise `None`.
    #[inline]
    pub fn check_positive_shape(self) -> Option<Self> {
        self.shape.is_positive().then(|| self)
    }

    /// Returns the extent containing only the points in both `self` and `other`.
    ///
    /// ```
    /// # use ilattice::extent::Extent;
    /// # use glam::IVec2;
    /// let e1 = Extent::from_min_and_max(IVec2::from([0; 2]), IVec2::from([3; 2]));
    /// let e2 = Extent::from_min_and_max(IVec2::from([2; 2]), IVec2::from([4; 2]));
    ///
    /// assert_eq!(e1.intersection(&e2), Extent::from_min_and_max(IVec2::from([2; 2]), IVec2::from([3; 2])));
    /// assert!(!e1.intersection(&e2).is_empty());
    ///
    /// let e1 = Extent::from_min_and_max(IVec2::from([0; 2]), IVec2::from([1; 2]));
    /// let e2 = Extent::from_min_and_max(IVec2::from([3; 2]), IVec2::from([4; 2]));
    ///
    /// assert_eq!(e1.intersection(&e2).shape, IVec2::from([0; 2]));
    /// assert!(e1.intersection(&e2).is_empty());
    /// ```
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        let minimum = self.minimum.least_upper_bound(other.minimum);
        let lub = self
            .least_upper_bound()
            .greatest_lower_bound(other.least_upper_bound());

        Self::from_min_and_lub(minimum, lub)
    }

    /// Returns the smallest extent containing all points in `self` or `other`.
    #[inline]
    pub fn bound_union(&self, other: &Self) -> Self {
        let minimum = self.minimum.greatest_lower_bound(other.minimum);
        let lub = self
            .least_upper_bound()
            .least_upper_bound(other.least_upper_bound());

        Self::from_min_and_lub(minimum, lub)
    }

    /// Returns `true` iff the intersection of `self` and `other` is equal to `self`.
    #[inline]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.intersection(other).eq(self)
    }

    /// Returns all 4 corners of a 2-dimensional extent.
    #[inline]
    pub fn corners2(&self) -> [V; 4]
    where
        V: Vector2,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();

        [
            V::from([min.x(), min.y()]),
            V::from([lub.x(), min.y()]),
            V::from([min.x(), lub.y()]),
            V::from([lub.x(), lub.y()]),
        ]
    }

    /// Returns all 8 corners of a 3-dimensional extent.
    #[inline]
    pub fn corners3(&self) -> [V; 8]
    where
        V: Vector3,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();

        [
            V::from([min.x(), min.y(), min.z()]),
            V::from([lub.x(), min.y(), min.z()]),
            V::from([min.x(), lub.y(), min.z()]),
            V::from([lub.x(), lub.y(), min.z()]),
            V::from([min.x(), min.y(), lub.z()]),
            V::from([lub.x(), min.y(), lub.z()]),
            V::from([min.x(), lub.y(), lub.z()]),
            V::from([lub.x(), lub.y(), lub.z()]),
        ]
    }

    #[inline]
    pub fn split2(&self, split: V) -> [Self; 4]
    where
        V: Vector2,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();

        [
            Self::from_min_and_lub(min, split),
            Self::from_min_and_lub(V::from([split.x(), min.y()]), V::from([lub.x(), split.y()])),
            Self::from_min_and_lub(V::from([min.x(), split.y()]), V::from([split.x(), lub.y()])),
            Self::from_min_and_lub(split, lub),
        ]
    }

    #[inline]
    pub fn split3(&self, split: V) -> [Self; 8]
    where
        V: Vector3,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();

        [
            Self::from_min_and_lub(min, split),
            Self::from_min_and_lub(
                V::from([split.x(), min.y(), min.z()]),
                V::from([lub.x(), split.y(), split.z()]),
            ),
            Self::from_min_and_lub(
                V::from([min.x(), split.y(), min.z()]),
                V::from([split.x(), lub.y(), split.z()]),
            ),
            Self::from_min_and_lub(
                V::from([split.x(), split.y(), min.z()]),
                V::from([lub.x(), lub.y(), split.z()]),
            ),
            Self::from_min_and_lub(
                V::from([min.x(), min.y(), split.z()]),
                V::from([split.x(), split.y(), lub.z()]),
            ),
            Self::from_min_and_lub(
                V::from([split.x(), min.y(), split.z()]),
                V::from([lub.x(), split.y(), lub.z()]),
            ),
            Self::from_min_and_lub(
                V::from([min.x(), split.y(), split.z()]),
                V::from([split.x(), lub.y(), lub.z()]),
            ),
            Self::from_min_and_lub(split, lub),
        ]
    }

    #[inline]
    pub fn split2_single(&self, split: V, quadrant: u8) -> Self
    where
        V: Vector2,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();
        let all_coords = [min.x(), min.y(), split.x(), split.y(), lub.x(), lub.y()];

        // Corresponds to the coordinate permutation in split2.
        const LUT: [[usize; 4]; 8] = [
            [0, 1, 2, 3],
            [2, 1, 4, 3],
            [0, 3, 2, 5],
            [2, 3, 4, 5],
            [0, 1, 2, 3],
            [2, 1, 4, 3],
            [0, 3, 2, 5],
            [2, 3, 4, 5],
        ];
        let [mx, my, lx, ly] = LUT[quadrant as usize].map(|i| all_coords[i]);

        Self::from_min_and_lub(V::from([mx, my]), V::from([lx, ly]))
    }

    #[inline]
    pub fn split3_single(&self, split: V, octant: u8) -> Self
    where
        V: Vector3,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();
        let all_coords = [
            min.x(),
            min.y(),
            min.z(),
            split.x(),
            split.y(),
            split.z(),
            lub.x(),
            lub.y(),
            lub.z(),
        ];

        // Corresponds to the coordinate permutation in split3.
        const LUT: [[usize; 6]; 8] = [
            [0, 1, 2, 3, 4, 5],
            [3, 1, 2, 6, 4, 5],
            [0, 4, 2, 3, 7, 5],
            [3, 4, 2, 6, 7, 5],
            [0, 1, 5, 3, 4, 8],
            [3, 1, 5, 6, 4, 8],
            [0, 4, 5, 3, 7, 8],
            [3, 4, 5, 6, 7, 8],
        ];
        let [mx, my, mz, lx, ly, lz] = LUT[octant as usize].map(|i| all_coords[i]);

        Self::from_min_and_lub(V::from([mx, my, mz]), V::from([lx, ly, lz]))
    }

    pub fn surface_area3(&self) -> V::Scalar
    where
        V: Vector3,
    {
        (V::Scalar::ONE + V::Scalar::ONE)
            * (self.shape.x() * self.shape.y()
                + self.shape.y() * self.shape.z()
                + self.shape.z() * self.shape.x())
    }
}

impl<V> Extent<V>
where
    V: IntegerVector,
{
    /// An alternative representation of an integer extent as the minimum point and maximum point. This only works for integer
    /// extents, where there is a unique maximum point.
    #[inline]
    pub fn from_min_and_max(minimum: V, max: V) -> Self {
        Self::from_min_and_lub(minimum, max + V::ONES)
    }

    /// Constructs the unique extent with both `p1` and `p2` as corners.
    #[inline]
    pub fn from_corners(p1: V, p2: V) -> Self {
        let min = p1.greatest_lower_bound(p2);
        let max = p1.least_upper_bound(p2);

        Self::from_min_and_max(min, max)
    }

    /// The number of points contained in the extent.
    #[inline]
    pub fn num_points(&self) -> u64 {
        let volume = self.volume();
        match volume.try_into() {
            Ok(n) => n,
            Err(_) => panic!("Failed to convert {:?} to u64", volume),
        }
    }

    /// Returns `true` iff `self.num_points() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_points() == 0
    }

    /// The unique greatest point in the extent.
    #[inline]
    pub fn max(&self) -> V {
        let lub = self.least_upper_bound();

        lub - V::ONES
    }

    /// Clamps `v` to force in **inside** of the `self` extent.
    ///
    /// ```
    /// # use ilattice::extent::Extent;
    /// # use glam::IVec2;
    /// let e = Extent::from_min_and_max(IVec2::new(-1, 5), IVec2::new(2, 10));
    /// let p_in = IVec2::new(0, 8);
    /// let p_out = IVec2::new(-4, 20);
    ///
    /// assert_eq!(e.clamp(p_in), p_in);
    /// assert_eq!(e.clamp(p_out), IVec2::new(-1, 10));
    /// ```
    #[inline]
    pub fn clamp(&self, v: V) -> V {
        v.least_upper_bound(self.minimum)
            .greatest_lower_bound(self.max())
    }

    /// Returns an iterator over all points in this 2-dimensional extent.
    ///
    /// ```
    /// # use ilattice::extent::Extent;
    /// # use glam::UVec2;
    /// let e = Extent::from_min_and_shape(UVec2::new(1, 2), UVec2::new(2, 2));
    ///
    /// let points: Vec<_> = e.iter2().collect();
    ///
    /// assert_eq!(
    ///     points,
    ///     vec![
    ///         UVec2::new(1, 2),
    ///         UVec2::new(2, 2),
    ///         UVec2::new(1, 3),
    ///         UVec2::new(2, 3)
    ///     ]
    /// );
    /// ```
    #[inline]
    pub fn iter2(&self) -> impl Iterator<Item = V>
    where
        V: Vector2,
        std::ops::Range<V::IntScalar>: Iterator<Item = V::IntScalar>,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();
        let y_range = min.y()..lub.y();
        let x_range = min.x()..lub.x();

        y_range.flat_map(move |y| x_range.clone().map(move |x| V::from([x, y])))
    }

    #[cfg(feature = "rayon")]
    /// Returns a rayon parallel iterator over all points in this 2-dimensional extent.
    ///
    /// ```
    /// # use ilattice::extent::Extent;
    /// # use rayon::prelude::*;
    /// # use glam::UVec2;
    /// let e = Extent::from_min_and_shape(UVec2::new(1, 2), UVec2::new(2, 2));
    ///
    /// let points: Vec<_> = e.par_iter2().collect();
    ///
    /// assert_eq!(
    ///     points,
    ///     vec![
    ///         UVec2::new(1, 2),
    ///         UVec2::new(2, 2),
    ///         UVec2::new(1, 3),
    ///         UVec2::new(2, 3)
    ///     ]
    /// );
    /// ```
    #[inline]
    pub fn par_iter2(&self) -> impl ParallelIterator<Item = V>
    where
        V: Vector2 + Send,
        V::Scalar: Send + Sync,
        std::ops::Range<V::IntScalar>: IntoParallelIterator<Item = V::IntScalar>,
    {
        let min_x = self.minimum.x();
        let lub_x = self.least_upper_bound().x();

        (self.minimum.y()..self.least_upper_bound().y())
            .into_par_iter()
            .flat_map(move |y| (min_x..lub_x).into_par_iter().map(move |x| V::from([x, y])))
    }

    /// Returns an iterator over all points in this 3-dimensional extent.
    /// ```
    /// # use ilattice::extent::Extent;
    /// # use glam::UVec3;
    /// let e = Extent::from_min_and_shape(UVec3::new(1, 2, 3), UVec3::new(2, 2, 2));
    ///
    /// let points: Vec<_> = e.iter3().collect();
    ///
    /// assert_eq!(
    ///     points,
    ///     vec![
    ///         UVec3::new(1, 2, 3),
    ///         UVec3::new(2, 2, 3),
    ///         UVec3::new(1, 3, 3),
    ///         UVec3::new(2, 3, 3),
    ///         UVec3::new(1, 2, 4),
    ///         UVec3::new(2, 2, 4),
    ///         UVec3::new(1, 3, 4),
    ///         UVec3::new(2, 3, 4)
    ///     ]
    /// );
    /// ```
    #[inline]
    pub fn iter3(&self) -> impl Iterator<Item = V>
    where
        V: Vector3,
        std::ops::Range<V::IntScalar>: Iterator<Item = V::IntScalar>,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();
        let z_range = min.z()..lub.z();
        let y_range = min.y()..lub.y();
        let x_range = min.x()..lub.x();

        z_range.flat_map(move |z| {
            y_range.clone().flat_map({
                let x_range = x_range.clone();
                move |y| x_range.clone().map(move |x| V::from([x, y, z]))
            })
        })
    }

    #[cfg(feature = "rayon")]
    /// Returns a rayon parallel iterator over all points in this 3-dimensional extent.
    /// ```
    /// # use ilattice::extent::Extent;
    /// # use rayon::prelude::*;
    /// # use glam::UVec3;
    /// let e = Extent::from_min_and_shape(UVec3::new(1, 2, 3), UVec3::new(2, 2, 2));
    ///
    /// let points: Vec<_> = e.par_iter3().collect();
    ///
    /// assert_eq!(
    ///     points,
    ///     vec![
    ///         UVec3::new(1, 2, 3),
    ///         UVec3::new(2, 2, 3),
    ///         UVec3::new(1, 3, 3),
    ///         UVec3::new(2, 3, 3),
    ///         UVec3::new(1, 2, 4),
    ///         UVec3::new(2, 2, 4),
    ///         UVec3::new(1, 3, 4),
    ///         UVec3::new(2, 3, 4)
    ///     ]
    /// );
    /// ```
    #[inline]
    pub fn par_iter3(&self) -> impl ParallelIterator<Item = V>
    where
        V: Vector3 + Send,
        V::Scalar: Send + Sync,
        std::ops::Range<V::IntScalar>: IntoParallelIterator<Item = V::IntScalar>,
    {
        let lub = self.least_upper_bound();
        let min_y = self.minimum.y();
        let lub_y = lub.y();
        let min_x = self.minimum.x();
        let lub_x = lub.x();

        (self.minimum.z()..lub.z())
            .into_par_iter()
            .flat_map(move |z| {
                (min_y..lub_y).into_par_iter().flat_map(move |y| {
                    (min_x..lub_x)
                        .into_par_iter()
                        .map(move |x| V::from([x, y, z]))
                })
            })
    }

    /// Returns the smallest extent containing all of the given points.
    #[inline]
    pub fn bound_points<I>(mut points: I) -> Self
    where
        I: Iterator<Item = V>,
    {
        let first_v = points
            .next()
            .expect("Cannot find bounding extent of empty set of points");

        let mut min_point = first_v;
        let mut max_point = first_v;
        for v in points {
            min_point = min_point.greatest_lower_bound(v);
            max_point = max_point.least_upper_bound(v);
        }

        Extent::from_min_and_max(min_point, max_point)
    }
}

impl<Vf> Extent<Vf>
where
    Vf: FloatVector,
{
    pub fn center(&self) -> Vf {
        let one = Vf::FloatScalar::ONE;
        self.minimum + self.shape / (one + one)
    }
}

impl<Vf, Vi> Extent<Vf>
where
    Vf: FloatVector<Int = Vi>,
    Vi: IntegerVector,
{
    /// Returns the integer `Extent` that contains `self`.
    #[inline]
    pub fn containing_integer_extent(&self) -> Extent<Vi> {
        Extent::from_min_and_max(
            self.minimum.floor().cast(),
            self.least_upper_bound().floor().cast(),
        )
    }
}

impl<V> Add<V> for Extent<V>
where
    V: Add<Output = V>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: V) -> Self::Output {
        Extent {
            minimum: self.minimum + rhs,
            shape: self.shape,
        }
    }
}

impl<V> Sub<V> for Extent<V>
where
    V: Sub<Output = V>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: V) -> Self::Output {
        Extent {
            minimum: self.minimum - rhs,
            shape: self.shape,
        }
    }
}

impl<V, Rhs> Mul<Rhs> for Extent<V>
where
    V: Mul<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Extent {
            minimum: self.minimum * rhs,
            shape: self.shape * rhs,
        }
    }
}

impl<V, Rhs> Shl<Rhs> for Extent<V>
where
    V: Shl<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn shl(self, rhs: Rhs) -> Self::Output {
        Extent {
            minimum: self.minimum << rhs,
            shape: self.shape << rhs,
        }
    }
}

impl<V, Rhs> Shr<Rhs> for Extent<V>
where
    V: Shr<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: Rhs) -> Self::Output {
        Extent {
            minimum: self.minimum >> rhs,
            shape: self.shape >> rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::*;

    #[test]
    fn splits_are_consistent() {
        let e = Extent::from_min_and_lub(Vec2::new(0.0, 1.0), Vec2::new(3.0, 4.0));
        let split_at = Vec2::new(2.0, 3.0);

        assert_eq!(
            e.split2(split_at),
            [0, 1, 2, 3].map(|quadrant| e.split2_single(split_at, quadrant))
        );

        let e = Extent::from_min_and_lub(Vec3::new(0.0, 1.0, 2.0), Vec3::new(6.0, 7.0, 8.0));
        let split_at = Vec3::new(3.0, 4.0, 5.0);

        assert_eq!(
            e.split3(split_at),
            [0, 1, 2, 3, 4, 5, 6, 7].map(|octant| e.split3_single(split_at, octant))
        );
    }
}
