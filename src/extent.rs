use crate::vector::*;

use core::ops::{Add, Mul, Shl, Shr, Sub};
use std::convert::TryInto;

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
    /// # use glam::const_ivec2;
    /// let e1 = Extent::from_min_and_max(const_ivec2!([0; 2]), const_ivec2!([1; 2]));
    /// let e2 = Extent::from_min_and_max(const_ivec2!([3; 2]), const_ivec2!([4; 2]));
    ///
    /// assert_eq!(e1.intersection(&e2).shape, const_ivec2!([0; 2]));
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

    /// Returns the smallest extent containing all the points in `self` or `other`.
    ///
    /// This is not strictly a union of sets of points, but it is the closest we can get while still using an `Extent`.
    #[inline]
    pub fn quasi_union(&self, other: &Self) -> Self {
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
    pub fn num_points(&self) -> usize {
        match self.volume().try_into() {
            Ok(n) => n,
            Err(_) => panic!("Negative volume!"),
        }
    }

    /// Returns `true` iff the number of points in the extent is 0.
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

    /// Calls `f` on every point in `self`.
    #[inline]
    pub fn for_each2(&self, mut f: impl FnMut(V))
    where
        V: Vector2,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();
        let mut y = min.y();
        while y < lub.y() {
            let mut x = min.x();
            while x < lub.x() {
                f(V::from([x, y]));
                x = x + V::IntScalar::ONE;
            }
            y = y + V::IntScalar::ONE;
        }
    }

    /// Calls `f` on every point in `self`.
    #[inline]
    pub fn for_each3(&self, mut f: impl FnMut(V))
    where
        V: Vector3,
    {
        let min = self.minimum;
        let lub = self.least_upper_bound();
        let mut z = min.z();
        while z < lub.z() {
            let mut y = min.y();
            while y < lub.y() {
                let mut x = min.x();
                while x < lub.x() {
                    f(V::from([x, y, z]));
                    x = x + V::IntScalar::ONE;
                }
                y = y + V::IntScalar::ONE;
            }
            z = z + V::IntScalar::ONE;
        }
    }

    /// Returns the smallest extent containing all of the given vectors.
    #[inline]
    pub fn bounding_extent<I>(mut vectors: I) -> Self
    where
        I: Iterator<Item = V>,
    {
        let first_v = vectors
            .next()
            .expect("Cannot find bounding extent of empty set of vectors");

        let mut min_point = first_v;
        let mut max_point = first_v;
        for v in vectors {
            min_point = min_point.greatest_lower_bound(v);
            max_point = max_point.least_upper_bound(v);
        }

        Extent::from_min_and_max(min_point, max_point)
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
            self.minimum.floor().cast_int(),
            self.least_upper_bound().floor().cast_int(),
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
