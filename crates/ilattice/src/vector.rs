//! Traits for implementing generic types.

use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::{
    fmt::Debug,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

/// A scalar value type to be used with [`Vector`].
pub trait Scalar:
    'static
    + Sized
    + Copy
    + Debug
    + Zero
    + One
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + PartialOrd
    + RangeMax
{
}

/// Primitive casts, including vectorized casts.
#[allow(missing_docs)]
pub trait PrimitiveCast<T> {
    fn cast(self) -> T;
}

/// A vector over the scalar field `Self::Scalar`.
#[allow(missing_docs)]
pub trait Vector:
    'static
    + Sized
    + Copy
    + Splat<Self::Scalar>
    + Map<Self::Scalar>
    + ZipMap<Self::Scalar>
    + Fold<Self::Scalar>
    + Ones
    + Zero
    + VectorArithmetic<Self::Scalar>
    + LatticeOrder
    + Min<Self::Scalar>
    + Max<Self::Scalar>
    + PartialEq
{
    type Scalar: Scalar;

    #[inline]
    fn is_positive(self) -> bool {
        self.with_lattice_ord() > Self::ZERO.with_lattice_ord()
    }
}

/// A 2-dimensional [`Vector`].
#[allow(missing_docs)]
pub trait Vector2: Vector + From<[Self::Scalar; 2]> {
    fn x(self) -> Self::Scalar;
    fn y(self) -> Self::Scalar;

    fn x_mut(&mut self) -> &mut Self::Scalar;
    fn y_mut(&mut self) -> &mut Self::Scalar;

    #[inline]
    fn as_array<T>(self) -> [T; 2]
    where
        Self::Scalar: PrimitiveCast<T>,
    {
        [self.x().cast(), self.y().cast()]
    }
}

/// A 3-dimensional [`Vector`].
#[allow(missing_docs)]
pub trait Vector3: Vector + From<[Self::Scalar; 3]> {
    fn x(self) -> Self::Scalar;
    fn y(self) -> Self::Scalar;
    fn z(self) -> Self::Scalar;

    fn x_mut(&mut self) -> &mut Self::Scalar;
    fn y_mut(&mut self) -> &mut Self::Scalar;
    fn z_mut(&mut self) -> &mut Self::Scalar;

    #[inline]
    fn as_array<T>(self) -> [T; 3]
    where
        Self::Scalar: PrimitiveCast<T>,
    {
        [self.x().cast(), self.y().cast(), self.z().cast()]
    }
}

/// Arithmetic operations on [`Vector`]s.
#[allow(missing_docs)]
pub trait VectorArithmetic<T>:
    Sized
    + Add<T, Output = Self>
    + Add<Output = Self>
    + Div<T, Output = Self>
    + Div<Self, Output = Self>
    + Mul<T, Output = Self>
    + Mul<Self, Output = Self>
    + Rem<T, Output = Self>
    + Rem<Self, Output = Self>
    + Sub<T, Output = Self>
    + Sub<Output = Self>
{
}

/// Creates a vector with all components equal to `value`.
#[allow(missing_docs)]
pub trait Splat<T> {
    fn splat(value: T) -> Self;
}

/// Applies `f` to all components, returning the results as `Self`.
#[allow(missing_docs)]
pub trait Map<T> {
    fn map(self, f: impl Fn(T) -> T) -> Self;
}

/// Zips the components of `self` and `other`, applying `f`, and returning the
/// results as `Self`.
#[allow(missing_docs)]
pub trait ZipMap<T> {
    fn zip_map(self, other: Self, f: impl Fn(T, T) -> T) -> Self;
}

/// Folds `f` over the vector components onto `init`.
#[allow(missing_docs)]
pub trait Fold<T> {
    fn fold<Out>(self, init: Out, f: impl Fn(T, Out) -> Out) -> Out;
}

/// Returns the least component.
#[allow(missing_docs)]
pub trait Min<T> {
    fn min_element(self) -> T;
}

/// Returns the greatest component.
#[allow(missing_docs)]
pub trait Max<T> {
    fn max_element(self) -> T;
}

/// Defines the constant zero scalar/vector.
#[allow(missing_docs)]
pub trait Zero {
    const ZERO: Self;
}

/// Defines the constant one scalar.
#[allow(missing_docs)]
pub trait One {
    const ONE: Self;
}

/// Defines the constant vector of all ones.
#[allow(missing_docs)]
pub trait Ones {
    const ONES: Self;
}

/// A trait denoting that the `PartialOrd` for `Self::LatticeVector` is
/// consistent with the [lattice](https://en.wikipedia.org/wiki/Lattice_(order))
/// structure.
#[allow(missing_docs)]
pub trait LatticeOrder {
    type LatticeVector: PartialOrd;

    fn with_lattice_ord(self) -> Self::LatticeVector;

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

/// A newtype that can be used to override the `PartialOrd` implementation of
/// `T` so that it is consistent with `LatticeOrder`.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WithLatticeOrd<T>(pub T);

/// Defines the constant bounds of a scalar/vector types.
#[allow(missing_docs)]
pub trait Bounded {
    const MIN: Self;
    const MAX: Self;
}

/// Given some scalar, define how it defines the maximum value over a range.
///
/// This works differently for integers and real numbers.
#[allow(missing_docs)]
pub trait RangeMax {
    fn range_max(min: Self, length: Self) -> Self;
    fn range_length(min: Self, max: Self) -> Self;
}

mod signed_vector {
    use super::{Neg, Vector};

    /// Returns the sign of each vector component.
    #[allow(missing_docs)]
    pub trait SignedVector: Vector + Abs + Neg {
        fn signum(self) -> Self;
    }

    /// Returns the absolute value of each vector component.
    #[allow(missing_docs)]
    pub trait Abs {
        fn abs(self) -> Self;
    }
}
pub use signed_vector::*;

mod integer_vector {
    use super::{PrimitiveCast, Scalar, Vector};

    use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
    use std::convert::TryInto;

    /// A [`Scalar`] with integer-specific behaviors.
    #[allow(missing_docs)]
    pub trait IntegerScalar:
        Scalar
        + TryInto<u64>
        + PrimitiveCast<u8>
        + PrimitiveCast<u16>
        + PrimitiveCast<u32>
        + PrimitiveCast<u64>
    {
        fn is_power_of_two(self) -> bool;
        fn trailing_zeros(self) -> Self;
    }

    /// A `Vector<T>` where `T` is some integer.
    ///
    /// This enables `Eq`, bitwise logical operations, and bit shifting.
    #[allow(missing_docs)]
    pub trait IntegerVector:
        Vector<Scalar = Self::IntScalar>
        + Eq
        + ScalarBitwiseLogic<Self::Scalar>
        + VectorBitwiseLogic
        + AllShiftOps<Self::IntScalar>
        + PrimitiveCast<Self::Float>
    {
        type IntScalar: IntegerScalar;
        type Float;

        #[inline]
        fn all_dimensions_are_powers_of_two(self) -> bool {
            self.fold(true, |c, out| out && c.is_power_of_two())
        }
    }

    /// Bitwise operations for [`Scalar`]s.
    #[allow(missing_docs)]
    pub trait ScalarBitwiseLogic<Rhs>:
        Sized + BitAnd<Rhs, Output = Self> + BitOr<Rhs, Output = Self> + BitXor<Rhs, Output = Self>
    {
    }

    /// Bitwise operations for [`Vector`]s.
    #[allow(missing_docs)]
    pub trait VectorBitwiseLogic:
        Sized
        + BitAnd<Self, Output = Self>
        + BitOr<Self, Output = Self>
        + BitXor<Self, Output = Self>
        + Not<Output = Self>
    {
    }

    /// Bit shifting operations for [`Scalar`]s and [`Vector`]s.
    #[allow(missing_docs)]
    pub trait ShiftOps<Rhs>: Sized + Shl<Rhs, Output = Self> + Shr<Rhs, Output = Self> {}

    /// Bit shifting with any unsigned integer or `Self` RHS operand.
    ///
    /// Signed shifting has surprising behavior. HOWEVER we do support shifting
    /// by `Self` as it makes generic code much simpler.
    #[allow(missing_docs)]
    pub trait AllShiftOps<T>:
        ShiftOps<Self>
        + ShiftOps<T>
        + ShiftOps<Self::UintVec>
        + ShiftOps<u8>
        + ShiftOps<u16>
        + ShiftOps<u32>
    {
        type UintVec;
    }
}
pub use integer_vector::*;

mod float_vector {
    use super::{PrimitiveCast, Scalar, Vector};

    /// A `Vector<T>` where `T` is some floating point number.
    #[allow(missing_docs)]
    pub trait FloatVector:
        Vector<Scalar = Self::FloatScalar> + RoundingOps + PrimitiveCast<Self::Int>
    {
        type FloatScalar: Scalar;
        type Int;
    }

    /// Rounding operations for [`FloatVector`]s.
    #[allow(missing_docs)]
    pub trait RoundingOps {
        fn floor(self) -> Self;
        fn ceil(self) -> Self;
    }
}
pub use float_vector::*;

mod scalar_impl {
    use super::{Bounded, IntegerScalar, One, PrimitiveCast, RangeMax, Scalar, Zero};

    macro_rules! impl_integer_scalar {
        ($t:ident) => {
            impl Scalar for $t {}

            impl IntegerScalar for $t {
                #[inline]
                fn is_power_of_two(self) -> bool {
                    PrimitiveCast::<u32>::cast(self).is_power_of_two()
                }
                #[inline]
                fn trailing_zeros(self) -> Self {
                    self.trailing_zeros() as $t
                }
            }

            impl PrimitiveCast<u8> for $t {
                #[inline]
                fn cast(self) -> u8 {
                    self as u8
                }
            }
            impl PrimitiveCast<u16> for $t {
                #[inline]
                fn cast(self) -> u16 {
                    self as u16
                }
            }
            impl PrimitiveCast<u32> for $t {
                #[inline]
                fn cast(self) -> u32 {
                    self as u32
                }
            }
            impl PrimitiveCast<u64> for $t {
                #[inline]
                fn cast(self) -> u64 {
                    self as u64
                }
            }

            impl Zero for $t {
                const ZERO: $t = 0;
            }
            impl One for $t {
                const ONE: $t = 1;
            }

            impl RangeMax for $t {
                #[inline]
                fn range_max(min: Self, length: Self) -> Self {
                    min + length - 1
                }
                #[inline]
                fn range_length(min: Self, max: Self) -> Self {
                    1 + max - min
                }
            }
        };
    }

    macro_rules! impl_float_scalar {
        ($t:ty) => {
            impl Scalar for $t {}

            impl Zero for $t {
                const ZERO: $t = 0.0;
            }
            impl One for $t {
                const ONE: $t = 1.0;
            }
            impl RangeMax for $t {
                #[inline]
                fn range_max(min: Self, length: Self) -> Self {
                    min + length
                }
                #[inline]
                fn range_length(min: Self, max: Self) -> Self {
                    max - min
                }
            }
        };
    }

    impl_integer_scalar!(u8);
    impl_integer_scalar!(u16);
    impl_integer_scalar!(u32);
    impl_integer_scalar!(i8);
    impl_integer_scalar!(i16);
    impl_integer_scalar!(i32);

    impl_float_scalar!(f32);
    impl_float_scalar!(f64);

    macro_rules! impl_bounded {
        ($t:ident) => {
            impl Bounded for $t {
                const MIN: Self = $t::MIN;
                const MAX: Self = $t::MAX;
            }
        };
    }

    impl_bounded!(u8);
    impl_bounded!(u16);
    impl_bounded!(u32);
    impl_bounded!(i8);
    impl_bounded!(i16);
    impl_bounded!(i32);
}
