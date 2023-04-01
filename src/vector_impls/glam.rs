use crate::vector::{Abs, AllShiftOps, Bounded, FloatVector, Fold, IntegerVector, LatticeOrder, Map, Max, Min, Ones, PrimitiveCast, RoundingOps, ScalarBitwiseLogic, ShiftOps, SignedVector, Splat, Vector, Vector2, Vector3, VectorArithmetic, VectorBitwiseLogic, WithLatticeOrd, Zero, ZipMap};

#[cfg(feature = "morton-encoding")]
use crate::morton::{EncodeMorton, Morton2i32, Morton2u32, Morton3i32, Morton3u32};

use core::cmp::Ordering;
use glam::{IVec2, IVec3, UVec2, UVec3, Vec2, Vec3, Vec3A};

macro_rules! impl_lattice_order {
    ($vec:ident, $scalar:ident) => {
        impl LatticeOrder for $vec {
            type LatticeVector = WithLatticeOrd<Self>;
            #[inline]
            fn with_lattice_ord(self) -> Self::LatticeVector {
                WithLatticeOrd(self)
            }
            #[inline]
            fn least_upper_bound(self, other: Self) -> Self {
                self.max(other)
            }
            #[inline]
            fn greatest_lower_bound(self, other: Self) -> Self {
                self.min(other)
            }
        }
        impl Min<$scalar> for $vec {
            #[inline]
            fn min_element(self) -> $scalar {
                self.min_element()
            }
        }
        impl Max<$scalar> for $vec {
            #[inline]
            fn max_element(self) -> $scalar {
                self.max_element()
            }
        }
    };
}

macro_rules! impl_integer_vec2 {
    ($ivec:ident, $fvec:ident, $fscalar:ident) => {
        impl PrimitiveCast<$fvec> for $ivec {
            #[inline]
            fn cast(self: Self) -> $fvec {
                $fvec::new(self.x as $fscalar, self.y as $fscalar)
            }
        }

        impl PartialOrd for WithLatticeOrd<$ivec> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                if self < other {
                    Some(Ordering::Less)
                } else if self > other {
                    Some(Ordering::Greater)
                } else if self.0.x == other.0.x && self.0.y == other.0.y {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }

            #[inline]
            fn lt(&self, other: &Self) -> bool {
                self.0.x < other.0.x && self.0.y < other.0.y
            }

            #[inline]
            fn gt(&self, other: &Self) -> bool {
                self.0.x > other.0.x && self.0.y > other.0.y
            }

            #[inline]
            fn le(&self, other: &Self) -> bool {
                self.0.x <= other.0.x && self.0.y <= other.0.y
            }

            #[inline]
            fn ge(&self, other: &Self) -> bool {
                self.0.x >= other.0.x && self.0.y >= other.0.y
            }
        }
    };
}

macro_rules! impl_integer_vec3 {
    ($ivec:ident, $fvec:ident, $fscalar:ident) => {
        impl PrimitiveCast<$fvec> for $ivec {
            #[inline]
            fn cast(self) -> $fvec {
                $fvec::new(self.x as $fscalar, self.y as $fscalar, self.z as $fscalar)
            }
        }

        impl PartialOrd for WithLatticeOrd<$ivec> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                if self < other {
                    Some(Ordering::Less)
                } else if self > other {
                    Some(Ordering::Greater)
                } else if self.0.x == other.0.x && self.0.y == other.0.y && self.0.z == other.0.z {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }

            #[inline]
            fn lt(&self, other: &Self) -> bool {
                self.0.x < other.0.x && self.0.y < other.0.y && self.0.z < other.0.z
            }

            #[inline]
            fn gt(&self, other: &Self) -> bool {
                self.0.x > other.0.x && self.0.y > other.0.y && self.0.z > other.0.z
            }

            #[inline]
            fn le(&self, other: &Self) -> bool {
                self.0.x <= other.0.x && self.0.y <= other.0.y && self.0.z <= other.0.z
            }

            #[inline]
            fn ge(&self, other: &Self) -> bool {
                self.0.x >= other.0.x && self.0.y >= other.0.y && self.0.z >= other.0.z
            }
        }
    };
}

macro_rules! impl_float_vec2_with_lattice_partial_ord {
    ($vec:ident) => {
        impl PartialOrd for WithLatticeOrd<$vec> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                if self < other {
                    Some(Ordering::Less)
                } else if self > other {
                    Some(Ordering::Greater)
                } else {
                    None
                }
            }

            #[inline]
            fn lt(&self, other: &Self) -> bool {
                self.0.x < other.0.x && self.0.y < other.0.y
            }

            #[inline]
            fn gt(&self, other: &Self) -> bool {
                self.0.x > other.0.x && self.0.y > other.0.y
            }

            #[inline]
            fn le(&self, other: &Self) -> bool {
                self.0.x <= other.0.x && self.0.y <= other.0.y
            }

            #[inline]
            fn ge(&self, other: &Self) -> bool {
                self.0.x >= other.0.x && self.0.y >= other.0.y
            }
        }
    };
}

macro_rules! impl_float_vec3_with_lattice_partial_ord {
    ($vec:ident) => {
        impl PartialOrd for WithLatticeOrd<$vec> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                if self < other {
                    Some(Ordering::Less)
                } else if self > other {
                    Some(Ordering::Greater)
                } else {
                    None
                }
            }

            #[inline]
            fn lt(&self, other: &Self) -> bool {
                self.0.x < other.0.x && self.0.y < other.0.y && self.0.z < other.0.z
            }

            #[inline]
            fn gt(&self, other: &Self) -> bool {
                self.0.x > other.0.x && self.0.y > other.0.y && self.0.z > other.0.z
            }

            #[inline]
            fn le(&self, other: &Self) -> bool {
                self.0.x <= other.0.x && self.0.y <= other.0.y && self.0.z <= other.0.z
            }

            #[inline]
            fn ge(&self, other: &Self) -> bool {
                self.0.x >= other.0.x && self.0.y >= other.0.y && self.0.z >= other.0.z
            }
        }
    };
}

macro_rules! impl_signed_shift_ops {
    ($vec:ident, $scalar:ident, $uvec:ident) => {
        impl AllShiftOps<$scalar> for $vec {
            type UintVec = $uvec;
        }
        impl ShiftOps<u8> for $vec {}
        impl ShiftOps<u16> for $vec {}
        impl ShiftOps<u32> for $vec {}
        impl ShiftOps<$scalar> for $vec {}
        impl ShiftOps<$vec> for $vec {}
        impl ShiftOps<$uvec> for $vec {}
    };
}

macro_rules! impl_unsigned_shift_ops {
    ($vec:ident, $scalar:ident) => {
        impl AllShiftOps<$scalar> for $vec {
            type UintVec = $vec;
        }
        impl ShiftOps<u8> for $vec {}
        impl ShiftOps<u16> for $vec {}
        impl ShiftOps<u32> for $vec {}
        impl ShiftOps<$vec> for $vec {}
    };
}

macro_rules! impl_signed_vector {
    ($vec:ident) => {
        impl SignedVector for $vec {
            #[inline]
            fn signum(self) -> Self {
                self.signum()
            }
        }
        impl Abs for $vec {
            #[inline]
            fn abs(self) -> Self {
                self.abs()
            }
        }
    };
}

macro_rules! impl_integer_vector {
    ($vec:ident, $dim:literal, $scalar:ident, $fvec:ident, $uvec:ident, $ones:expr) => {
        impl IntegerVector for $vec {
            type IntScalar = $scalar;
            type Float = $fvec;
        }
        impl Vector for $vec {
            type Scalar = $scalar;
        }
        impl VectorArithmetic<$scalar> for $vec {}
        impl ScalarBitwiseLogic<$scalar> for $vec {}
        impl VectorBitwiseLogic for $vec {}

        impl Splat<$scalar> for $vec {
            #[inline]
            fn splat(value: $scalar) -> Self {
                Self::splat(value)
            }
        }
        impl Zero for $vec {
            const ZERO: Self = Self::ZERO;
        }
        impl Ones for $vec {
            const ONES: Self = $ones;
        }
    };
}

macro_rules! impl_float_vector {
    ($vec:ident, $scalar:ident, $ivec:ident, $ones:expr) => {
        impl FloatVector for $vec {
            type FloatScalar = $scalar;
            type Int = $ivec;
        }
        impl Vector for $vec {
            type Scalar = $scalar;
        }
        impl VectorArithmetic<$scalar> for $vec {}
        impl Splat<$scalar> for $vec {
            #[inline]
            fn splat(value: $scalar) -> Self {
                Self::splat(value)
            }
        }
        impl Zero for $vec {
            const ZERO: Self = $vec::ZERO;
        }
        impl Ones for $vec {
            const ONES: Self = $ones;
        }
        impl RoundingOps for $vec {
            #[inline]
            fn floor(self) -> Self {
                self.floor()
            }
            #[inline]
            fn ceil(self) -> Self {
                self.ceil()
            }
        }
    };
}

macro_rules! impl_float_vec2 {
    ($vec:ident, $ivec:ident, $iscalar:ident) => {
        impl PrimitiveCast<$ivec> for $vec {
            #[inline]
            fn cast(self) -> $ivec {
                $ivec::new(self.x as $iscalar, self.y as $iscalar)
            }
        }
    };
}

macro_rules! impl_float_vec3 {
    ($vec:ident, $ivec:ident, $iscalar:ident) => {
        impl PrimitiveCast<$ivec> for $vec {
            #[inline]
            fn cast(self) -> $ivec {
                $ivec::new(self.x as $iscalar, self.y as $iscalar, self.z as $iscalar)
            }
        }
    };
}

macro_rules! impl_vec2 {
    ($vec:ident, $scalar:ident) => {
        impl Vector2 for $vec {
            #[inline]
            fn x(self) -> Self::Scalar {
                self.x
            }
            #[inline]
            fn y(self) -> Self::Scalar {
                self.y
            }
            #[inline]
            fn x_mut(&mut self) -> &mut Self::Scalar {
                &mut self.x
            }
            #[inline]
            fn y_mut(&mut self) -> &mut Self::Scalar {
                &mut self.y
            }
        }
        impl Fold<$scalar> for $vec {
            #[inline]
            fn fold<T>(self, init: T, f: impl Fn(<Self as Vector>::Scalar, T) -> T) -> T {
                let mut out = init;
                out = f(self.x, out);
                out = f(self.y, out);
                out
            }
        }
        impl Map<$scalar> for $vec {
            /// Applies `f` to all components, returning the results as `Self`.
            #[inline]
            fn map(self, f: impl Fn($scalar) -> $scalar) -> Self {
                Self::new(f(self.x), f(self.y))
            }
        }
        impl ZipMap<$scalar> for $vec {
            /// Zips the components of `self` and `other`, applying `f`, and returning the results as `Self`.
            #[inline]
            fn zip_map(self, other: Self, f: impl Fn($scalar, $scalar) -> $scalar) -> Self {
                Self::new(f(self.x, other.x), f(self.y, other.y))
            }
        }
    };
}

macro_rules! impl_vec3 {
    ($vec:ident, $scalar:ident) => {
        impl Vector3 for $vec {
            #[inline]
            fn x(self) -> Self::Scalar {
                self.x
            }
            #[inline]
            fn y(self) -> Self::Scalar {
                self.y
            }
            #[inline]
            fn z(self) -> Self::Scalar {
                self.z
            }
            #[inline]
            fn x_mut(&mut self) -> &mut Self::Scalar {
                &mut self.x
            }
            #[inline]
            fn y_mut(&mut self) -> &mut Self::Scalar {
                &mut self.y
            }
            #[inline]
            fn z_mut(&mut self) -> &mut Self::Scalar {
                &mut self.z
            }
        }
        impl Fold<$scalar> for $vec {
            #[inline]
            fn fold<T>(self, init: T, f: impl Fn(<Self as Vector>::Scalar, T) -> T) -> T {
                let mut out = init;
                out = f(self.x, out);
                out = f(self.y, out);
                out = f(self.z, out);
                out
            }
        }
        impl Map<$scalar> for $vec {
            /// Applies `f` to all components, returning the results as `Self`.
            #[inline]
            fn map(self, f: impl Fn($scalar) -> $scalar) -> Self {
                Self::new(f(self.x), f(self.y), f(self.z))
            }
        }
        impl ZipMap<$scalar> for $vec {
            /// Zips the components of `self` and `other`, applying `f`, and returning the results as `Self`.
            #[inline]
            fn zip_map(self, other: Self, f: impl Fn($scalar, $scalar) -> $scalar) -> Self {
                Self::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
            }
        }
    };
}

// IVec2
impl_vec2!(IVec2, i32);
impl_integer_vector!(IVec2, 2, i32, Vec2, UVec2, IVec2::from_array([1; 2]));
impl_signed_vector!(IVec2);
impl_signed_shift_ops!(IVec2, i32, UVec2);
impl_integer_vec2!(IVec2, Vec2, f32);
impl_lattice_order!(IVec2, i32);
impl Bounded for IVec2 {
    const MIN: Self = Self::from_array([i32::MIN; 2]);
    const MAX: Self = Self::from_array([i32::MAX; 2]);
}

// IVec3
impl_vec3!(IVec3, i32);
// Note: casting to Vec3A is preferred over Vec3
impl_integer_vector!(IVec3, 3, i32, Vec3A, UVec3, IVec3::from_array([1; 3]));
impl_signed_vector!(IVec3);
impl_signed_shift_ops!(IVec3, i32, UVec3);
// Note: casting from Vec3A is preferred over Vec3
impl_integer_vec3!(IVec3, Vec3A, f32);
impl_lattice_order!(IVec3, i32);
impl Bounded for IVec3 {
    const MIN: Self = Self::from_array([i32::MIN; 3]);
    const MAX: Self = Self::from_array([i32::MAX; 3]);
}

// UVec2
impl_vec2!(UVec2, u32);
impl_integer_vector!(UVec2, 2, u32, Vec2, UVec2, UVec2::from_array([1; 2]));
impl_unsigned_shift_ops!(UVec2, u32);
impl_integer_vec2!(UVec2, Vec2, f32);
impl_lattice_order!(UVec2, u32);
impl Bounded for UVec2 {
    const MIN: Self = Self::from_array([u32::MIN; 2]);
    const MAX: Self = Self::from_array([u32::MAX; 2]);
}

// UVec3
impl_vec3!(UVec3, u32);
// Note: casting to Vec3A is preferred over Vec3
impl_integer_vector!(UVec3, 3, u32, Vec3A, UVec3, UVec3::from_array([1; 3]));
impl_unsigned_shift_ops!(UVec3, u32);
// Note: casting from Vec3A is preferred over Vec3
impl_integer_vec3!(UVec3, Vec3A, f32);
impl_lattice_order!(UVec3, u32);
impl Bounded for UVec3 {
    const MIN: Self = Self::from_array([u32::MIN; 3]);
    const MAX: Self = Self::from_array([u32::MAX; 3]);
}

// Vec2
impl_vec2!(Vec2, f32);
impl_float_vector!(Vec2, f32, IVec2, Vec2::from_array([1.0; 2]));
impl_float_vec2!(Vec2, IVec2, i32);
impl_signed_vector!(Vec2);
impl_float_vec2_with_lattice_partial_ord!(Vec2);
impl_lattice_order!(Vec2, f32);
impl Bounded for Vec2 {
    const MIN: Self = Self::from_array([f32::MIN; 2]);
    const MAX: Self = Self::from_array([f32::MAX; 2]);
}

// Vec3
impl_vec3!(Vec3, f32);
impl_float_vector!(Vec3, f32, IVec3, Vec3::from_array([1.0; 3]));
impl_float_vec3!(Vec3, IVec3, i32);
impl_signed_vector!(Vec3);
impl_float_vec3_with_lattice_partial_ord!(Vec3);
impl_lattice_order!(Vec3, f32);
impl Bounded for Vec3 {
    const MIN: Self = Self::from_array([f32::MIN; 3]);
    const MAX: Self = Self::from_array([f32::MAX; 3]);
}

// Vec3A
impl_vec3!(Vec3A, f32);
impl_float_vector!(Vec3A, f32, IVec3, Vec3A::from_array([1.0; 3]));
impl_float_vec3!(Vec3A, IVec3, i32);
impl_signed_vector!(Vec3A);
impl_float_vec3_with_lattice_partial_ord!(Vec3A);
impl_lattice_order!(Vec3A, f32);
impl Bounded for Vec3A {
    const MIN: Self = Self::from_array([f32::MIN; 3]);
    const MAX: Self = Self::from_array([f32::MAX; 3]);
}

#[cfg(feature = "morton-encoding")]
mod impl_morton {
    use super::*;

    macro_rules! impl_encode_morton {
        ($vec:ident, $dim:literal, $scalar:ty, $morton:ident) => {
            impl EncodeMorton for $vec {
                type Morton = $morton;
            }
            impl From<$morton> for $vec {
                #[inline]
                fn from(m: $morton) -> Self {
                    Self::from(<[$scalar; $dim]>::from(m))
                }
            }
            impl From<$vec> for $morton {
                #[inline]
                fn from(v: $vec) -> $morton {
                    $morton::from(v.to_array())
                }
            }
        };
    }

    impl_encode_morton!(IVec2, 2, i32, Morton2i32);
    impl_encode_morton!(IVec3, 3, i32, Morton3i32);
    impl_encode_morton!(UVec2, 2, u32, Morton2u32);
    impl_encode_morton!(UVec3, 3, u32, Morton3u32);
}
