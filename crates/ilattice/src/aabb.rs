use crate::vector::{FloatVector, IntegerVector, One, RangeMax, Vector, Vector2, Vector3};
use core::ops::{Add, Mul, Shl, Shr, Sub};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An N-dimensional AABB.
///
/// This is mathematically the Cartesian product of a closed interval `[a, b]`
/// in each dimension. You can also just think of it as an axis-aligned box with
/// some minimum and maximum point.
#[derive(Debug)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::CheckBytes, rkyv::Serialize, rkyv::Deserialize),
    archive(as = "Self"),
    archive(bound(archive = "V: rkyv::Archive<Archived=V>"))
)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Aabb<V> {
    /// The minimum point contained in the AABB.
    pub min: V,
    /// The maximum point contained in the AABB.
    pub max: V,
}

// A few of these traits could be derived. But it seems that derive will not
// help the compiler infer trait bounds as well.
impl<V> Clone for Aabb<V>
where
    V: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            min: self.min.clone(),
            max: self.max.clone(),
        }
    }
}
impl<V> Copy for Aabb<V> where V: Copy {}
impl<V> PartialEq for Aabb<V>
where
    V: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.min.eq(&other.min) && self.max.eq(&other.max)
    }
}
impl<V> Eq for Aabb<V> where V: Eq {}

impl<V> Aabb<V> {
    pub const EDGES2: [[usize; 2]; 4] = [[0b00, 0b01], [0b00, 0b10], [0b01, 0b11], [0b10, 0b11]];

    pub const EDGES3: [[usize; 2]; 12] = [
        [0b000, 0b001],
        [0b000, 0b010],
        [0b000, 0b100],
        [0b001, 0b011],
        [0b001, 0b101],
        [0b010, 0b011],
        [0b010, 0b110],
        [0b100, 0b101],
        [0b100, 0b110],
        [0b110, 0b111],
        [0b101, 0b111],
        [0b011, 0b111],
    ];

    #[inline]
    pub fn map_components<T>(&self, f: impl Fn(&V) -> T) -> Aabb<T> {
        Aabb {
            min: f(&self.min),
            max: f(&self.max),
        }
    }
}

impl<V> Aabb<V>
where
    V: Vector,
{
    /// The default representation of an AABB as the minimum and maximum points.
    #[inline]
    pub const fn from_min_and_max(min: V, max: V) -> Self {
        Self { min, max }
    }

    /// An alternative representation of an AABB as the minimum point and shape.
    #[inline]
    pub fn from_min_and_shape(min: V, shape: V) -> Self {
        let max = min.zip_map(shape, V::Scalar::range_max);
        Self::from_min_and_max(min, max)
    }

    /// Translate the AABB such that it has `new_min` as it's new minimum.
    #[inline]
    pub fn translate_to_min(&self, new_min: V) -> Self {
        Self::from_min_and_shape(new_min, self.shape())
    }

    /// Resize the AABB such that it has `new_shape` as it's new shape.
    #[inline]
    pub fn with_shape(&self, new_shape: V) -> Self {
        Self::from_min_and_shape(self.min, new_shape)
    }

    /// The length of each dimension of the AABB.
    ///
    /// For real numbers, this is just `max - min`. For integers, we define the
    /// 1-dimensional length as the number of points in the range `[min, max]`,
    /// which is calculated as `1 + max - min`.
    #[inline]
    pub fn shape(&self) -> V {
        self.min
            .zip_map(self.max, V::Scalar::range_length)
            .max(V::ZERO)
    }

    /// The volume of the AABB.
    ///
    /// For real numbers, `(max - min)^D`, and for integers, the number of
    /// points in the AABB.
    #[inline]
    pub fn volume(&self) -> V::Scalar {
        self.shape().fold(V::Scalar::ONE, |c, out| c * out)
    }

    /// Returns `true` iff the point `p` is contained in this AABB.
    #[inline]
    pub fn contains(&self, p: V) -> bool {
        self.min.with_lattice_ord() <= p.with_lattice_ord()
            && p.with_lattice_ord() <= self.max.with_lattice_ord()
    }

    /// Returns a new AABB that's been padded on all borders by `pad_amount`.
    #[inline]
    pub fn padded(&self, pad_amount: V::Scalar) -> Self {
        Self::from_min_and_max(
            self.min - V::splat(pad_amount),
            self.max + V::splat(pad_amount),
        )
    }

    /// Returns `Some(self)` iff this AABB has a positive shape, otherise
    /// `None`.
    #[inline]
    pub fn check_positive_shape(self) -> Option<Self> {
        self.shape().is_positive().then_some(self)
    }

    /// Returns the AABB containing only the points in both `self` and
    /// `other`.
    ///
    /// ```
    /// # use ilattice::Aabb;
    /// # use glam::IVec2;
    /// let e1 = Aabb::from_min_and_max(IVec2::from([0; 2]), IVec2::from([3; 2]));
    /// let e2 = Aabb::from_min_and_max(IVec2::from([2; 2]), IVec2::from([4; 2]));
    ///
    /// assert_eq!(
    ///     e1.intersection(&e2),
    ///     Aabb::from_min_and_max(IVec2::from([2; 2]), IVec2::from([3; 2]))
    /// );
    /// assert!(!e1.intersection(&e2).is_empty());
    ///
    /// let e1 = Aabb::from_min_and_max(IVec2::from([0; 2]), IVec2::from([1; 2]));
    /// let e2 = Aabb::from_min_and_max(IVec2::from([3; 2]), IVec2::from([4; 2]));
    ///
    /// assert_eq!(e1.intersection(&e2).shape(), IVec2::from([0; 2]));
    /// assert!(e1.intersection(&e2).is_empty());
    /// ```
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        Self::from_min_and_max(min, max)
    }

    /// Returns the smallest AABB containing all points in `self` or `other`.
    #[inline]
    pub fn bound_union(&self, other: &Self) -> Self {
        let min = self.min.min(other.min);
        let max = self.max.max(other.max);
        Self::from_min_and_max(min, max)
    }

    /// Returns `true` iff the intersection of `self` and `other` is equal to
    /// `self`.
    #[inline]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.intersection(other).eq(self)
    }

    /// Returns all 4 corners of a 2-dimensional AABB.
    #[inline]
    pub fn corners2(&self) -> [V; 4]
    where
        V: Vector2,
    {
        let min = self.min;
        let max = self.max;

        [
            V::from([min.x(), min.y()]),
            V::from([max.x(), min.y()]),
            V::from([min.x(), max.y()]),
            V::from([max.x(), max.y()]),
        ]
    }

    /// Returns all 8 corners of a 3-dimensional AABB.
    #[inline]
    pub fn corners3(&self) -> [V; 8]
    where
        V: Vector3,
    {
        let min = self.min;
        let max = self.max;

        [
            V::from([min.x(), min.y(), min.z()]),
            V::from([max.x(), min.y(), min.z()]),
            V::from([min.x(), max.y(), min.z()]),
            V::from([max.x(), max.y(), min.z()]),
            V::from([min.x(), min.y(), max.z()]),
            V::from([max.x(), min.y(), max.z()]),
            V::from([min.x(), max.y(), max.z()]),
            V::from([max.x(), max.y(), max.z()]),
        ]
    }

    #[inline]
    pub fn split2(&self, split: V) -> [Self; 4]
    where
        V: Vector2,
    {
        let min = self.min;
        let max = self.max;

        [
            Self::from_min_and_max(min, split),
            Self::from_min_and_max(V::from([split.x(), min.y()]), V::from([max.x(), split.y()])),
            Self::from_min_and_max(V::from([min.x(), split.y()]), V::from([split.x(), max.y()])),
            Self::from_min_and_max(split, max),
        ]
    }

    #[inline]
    pub fn split3(&self, split: V) -> [Self; 8]
    where
        V: Vector3,
    {
        let min = self.min;
        let max = self.max;

        [
            Self::from_min_and_max(min, split),
            Self::from_min_and_max(
                V::from([split.x(), min.y(), min.z()]),
                V::from([max.x(), split.y(), split.z()]),
            ),
            Self::from_min_and_max(
                V::from([min.x(), split.y(), min.z()]),
                V::from([split.x(), max.y(), split.z()]),
            ),
            Self::from_min_and_max(
                V::from([split.x(), split.y(), min.z()]),
                V::from([max.x(), max.y(), split.z()]),
            ),
            Self::from_min_and_max(
                V::from([min.x(), min.y(), split.z()]),
                V::from([split.x(), split.y(), max.z()]),
            ),
            Self::from_min_and_max(
                V::from([split.x(), min.y(), split.z()]),
                V::from([max.x(), split.y(), max.z()]),
            ),
            Self::from_min_and_max(
                V::from([min.x(), split.y(), split.z()]),
                V::from([split.x(), max.y(), max.z()]),
            ),
            Self::from_min_and_max(split, max),
        ]
    }

    #[inline]
    pub fn split2_single(&self, split: V, quadrant: u8) -> Self
    where
        V: Vector2,
    {
        let min = self.min;
        let max = self.max;
        let all_coords = [min.x(), min.y(), split.x(), split.y(), max.x(), max.y()];

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

        Self::from_min_and_max(V::from([mx, my]), V::from([lx, ly]))
    }

    #[inline]
    pub fn split3_single(&self, split: V, octant: u8) -> Self
    where
        V: Vector3,
    {
        let min = self.min;
        let max = self.max;
        let all_coords = [
            min.x(),
            min.y(),
            min.z(),
            split.x(),
            split.y(),
            split.z(),
            max.x(),
            max.y(),
            max.z(),
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

        Self::from_min_and_max(V::from([mx, my, mz]), V::from([lx, ly, lz]))
    }

    #[allow(clippy::suspicious_operation_groupings)]
    #[inline]
    pub fn surface_area3(&self) -> V::Scalar
    where
        V: Vector3,
    {
        let s = self.shape();
        (V::Scalar::ONE + V::Scalar::ONE) * (s.x() * s.y() + s.y() * s.z() + s.z() * s.x())
    }
}

impl<V> Aabb<V>
where
    V: IntegerVector,
{
    /// Constructs the unique AABB with both `p1` and `p2` as corners.
    #[inline]
    pub fn from_corners(p1: V, p2: V) -> Self {
        let min = p1.min(p2);
        let max = p1.max(p2);
        Self::from_min_and_max(min, max)
    }

    /// The number of points contained in the AABB.
    #[inline]
    pub fn num_points(&self) -> u64 {
        let volume = self.volume();
        volume
            .try_into()
            .map_or_else(|_| panic!("Failed to convert {volume:?} to u64"), |n| n)
    }

    /// The number of points contained in the AABB. Doesn't `panic`
    #[inline]
    pub fn checked_num_points(&self) -> Option<u64> {
        let volume = self.volume();
        volume.try_into().ok()
    }

    /// Returns `true` iff `self.num_points() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_points() == 0
    }

    /// Clamps `v` to force in **inside** of the `self` AABB.
    ///
    /// ```
    /// # use ilattice::Aabb;
    /// # use glam::IVec2;
    /// let e = Aabb::from_min_and_max(IVec2::new(-1, 5), IVec2::new(2, 10));
    /// let p_in = IVec2::new(0, 8);
    /// let p_out = IVec2::new(-4, 20);
    ///
    /// assert_eq!(e.clamp(p_in), p_in);
    /// assert_eq!(e.clamp(p_out), IVec2::new(-1, 10));
    /// ```
    #[inline]
    pub fn clamp(&self, v: V) -> V {
        v.max(self.min).min(self.max)
    }

    /// Returns an iterator over all points in this 2-dimensional AABB.
    ///
    /// ```
    /// # use ilattice::Aabb;
    /// # use glam::UVec2;
    /// let e = Aabb::from_min_and_shape(UVec2::new(1, 2), UVec2::new(2, 2));
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
        std::ops::RangeInclusive<V::IntScalar>: Iterator<Item = V::IntScalar>,
    {
        let y_range = self.min.y()..=self.max.y();
        let x_range = self.min.x()..=self.max.x();

        y_range.flat_map(move |y| x_range.clone().map(move |x| V::from([x, y])))
    }

    #[cfg(feature = "rayon")]
    /// Returns a rayon parallel iterator over all points in this 2-dimensional
    /// AABB.
    ///
    /// ```
    /// # use ilattice::Aabb;
    /// # use rayon::prelude::*;
    /// # use glam::UVec2;
    /// let e = Aabb::from_min_and_shape(UVec2::new(1, 2), UVec2::new(2, 2));
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
        std::ops::RangeInclusive<V::IntScalar>: IntoParallelIterator<Item = V::IntScalar>,
    {
        let min_x = self.min.x();
        let max_x = self.max.x();

        (self.min.y()..=self.max.y())
            .into_par_iter()
            .flat_map(move |y| {
                (min_x..=max_x)
                    .into_par_iter()
                    .map(move |x| V::from([x, y]))
            })
    }

    /// Returns an iterator over all points in this 3-dimensional AABB.
    /// ```
    /// # use ilattice::Aabb;
    /// # use glam::UVec3;
    /// let e = Aabb::from_min_and_shape(UVec3::new(1, 2, 3), UVec3::new(2, 2, 2));
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
        std::ops::RangeInclusive<V::IntScalar>: Iterator<Item = V::IntScalar>,
    {
        let z_range = self.min.z()..=self.max.z();
        let y_range = self.min.y()..=self.max.y();
        let x_range = self.min.x()..=self.max.x();

        z_range.flat_map(move |z| {
            y_range.clone().flat_map({
                let x_range = x_range.clone();
                move |y| x_range.clone().map(move |x| V::from([x, y, z]))
            })
        })
    }

    #[cfg(feature = "rayon")]
    /// Returns a rayon parallel iterator over all points in this 3-dimensional
    /// AABB.
    ///
    /// ```
    /// # use ilattice::Aabb;
    /// # use rayon::prelude::*;
    /// # use glam::UVec3;
    /// let e = Aabb::from_min_and_shape(UVec3::new(1, 2, 3), UVec3::new(2, 2, 2));
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
        std::ops::RangeInclusive<V::IntScalar>: IntoParallelIterator<Item = V::IntScalar>,
    {
        let min_y = self.min.y();
        let max_y = self.max.y();
        let min_x = self.min.x();
        let max_x = self.max.x();

        (self.min.z()..=self.max.z())
            .into_par_iter()
            .flat_map(move |z| {
                (min_y..=max_y).into_par_iter().flat_map(move |y| {
                    (min_x..=max_x)
                        .into_par_iter()
                        .map(move |x| V::from([x, y, z]))
                })
            })
    }

    /// Returns the smallest AABB containing all of the given points.
    #[inline]
    pub fn bound_points<I>(mut points: I) -> Self
    where
        I: Iterator<Item = V>,
    {
        let first_v = points
            .next()
            .expect("Cannot find bounding AABB of empty set of points");

        let mut min_point = first_v;
        let mut max_point = first_v;
        for v in points {
            min_point = min_point.min(v);
            max_point = max_point.max(v);
        }

        Self::from_min_and_max(min_point, max_point)
    }
}

impl<Vf> Aabb<Vf>
where
    Vf: FloatVector,
{
    #[inline]
    pub fn center(&self) -> Vf {
        let one = Vf::FloatScalar::ONE;
        (self.min + self.max) / (one + one)
    }
}

impl<Vf, Vi> Aabb<Vf>
where
    Vf: FloatVector<Int = Vi>,
    Vi: IntegerVector,
{
    /// Returns the integer `Aabb` that contains `self`.
    #[inline]
    pub fn containing_integer_aabb(&self) -> Aabb<Vi> {
        Aabb::from_min_and_max(self.min.floor().cast(), self.max.floor().cast())
    }
}

impl<V, Rhs> Add<Rhs> for Aabb<V>
where
    V: Add<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Rhs) -> Self::Output {
        Self {
            min: self.min + rhs,
            max: self.max + rhs,
        }
    }
}

impl<V, Rhs> Sub<Rhs> for Aabb<V>
where
    V: Sub<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Rhs) -> Self::Output {
        Self {
            min: self.min - rhs,
            max: self.max - rhs,
        }
    }
}

impl<V, Rhs> Mul<Rhs> for Aabb<V>
where
    V: Mul<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Self {
            min: self.min * rhs,
            max: self.max * rhs,
        }
    }
}

impl<V, Rhs> Shl<Rhs> for Aabb<V>
where
    V: Shl<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn shl(self, rhs: Rhs) -> Self::Output {
        Self {
            min: self.min << rhs,
            max: self.max << rhs,
        }
    }
}

impl<V, Rhs> Shr<Rhs> for Aabb<V>
where
    V: Shr<Rhs, Output = V>,
    Rhs: Copy,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: Rhs) -> Self::Output {
        Self {
            min: self.min >> rhs,
            max: self.max >> rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    #[test]
    fn splits_are_consistent() {
        let e = Aabb::from_min_and_max(Vec2::new(0.0, 1.0), Vec2::new(3.0, 4.0));
        let split_at = Vec2::new(2.0, 3.0);

        assert_eq!(
            e.split2(split_at),
            [0, 1, 2, 3].map(|quadrant| e.split2_single(split_at, quadrant))
        );

        let e = Aabb::from_min_and_max(Vec3::new(0.0, 1.0, 2.0), Vec3::new(6.0, 7.0, 8.0));
        let split_at = Vec3::new(3.0, 4.0, 5.0);

        assert_eq!(
            e.split3(split_at),
            [0, 1, 2, 3, 4, 5, 6, 7].map(|octant| e.split3_single(split_at, octant))
        );
    }
}
