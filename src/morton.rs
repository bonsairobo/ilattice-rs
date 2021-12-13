//! Morton Codes (Z curve)
//!
//! The canonical encoding scheme is `0b...zyxzyx` because this is similar to the order in which we would traverse an array (x
//! first, then y, then z).
//!
//! ```
//! # use ilattice::prelude::{Morton3u32, UVec3};
//! assert_eq!(Morton3u32::from(UVec3::new(1, 0, 0)), Morton3u32(0b0001));
//! assert_eq!(Morton3u32::from(UVec3::new(0, 1, 0)), Morton3u32(0b0010));
//! assert_eq!(Morton3u32::from(UVec3::new(0, 0, 1)), Morton3u32(0b0100));
//! ```

// NOTE: The morton-encoding crate interprets arrays as [z, y, x], but we reverse that order to [x, y, z].

pub trait EncodeMorton: From<Self::Morton> + Into<Self::Morton> {
    /// The Morton code (Z order) for this vector.
    type Morton;
}

mod impl_unsigned {
    use morton_encoding::{morton_decode, morton_encode};

    #[cfg(feature = "rkyv")]
    use rkyv::{Archive, Deserialize, Serialize};

    macro_rules! impl_unsigned_morton2 {
        ($morton:ident, $store:ident, $scalar:ident) => {
            #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
            #[cfg_attr(
                feature = "rkyv",
                derive(Archive, Deserialize, Serialize),
                archive_attr(derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord))
            )]
            pub struct $morton(pub $store);

            impl From<$morton> for [$scalar; 2] {
                #[inline]
                fn from(m: $morton) -> Self {
                    let [y, x] = morton_decode(m.0);
                    [x, y]
                }
            }
            impl From<[$scalar; 2]> for $morton {
                #[inline]
                fn from([x, y]: [$scalar; 2]) -> $morton {
                    $morton(morton_encode([y, x]))
                }
            }
        };
    }

    macro_rules! impl_unsigned_morton3 {
        ($morton:ident, $store:ident, $scalar:ident) => {
            #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
            #[cfg_attr(
                feature = "rkyv",
                derive(Archive, Deserialize, Serialize),
                archive_attr(derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord))
            )]
            pub struct $morton(pub $store);

            impl From<$morton> for [$scalar; 3] {
                #[inline]
                fn from(m: $morton) -> Self {
                    let [z, y, x] = morton_decode(m.0);
                    [x, y, z]
                }
            }
            impl From<[$scalar; 3]> for $morton {
                #[inline]
                fn from([x, y, z]: [$scalar; 3]) -> $morton {
                    $morton(morton_encode([z, y, x]))
                }
            }
        };
    }

    impl_unsigned_morton2!(Morton2u8, u16, u8);
    impl_unsigned_morton2!(Morton2u16, u32, u16);
    impl_unsigned_morton2!(Morton2u32, u64, u32);

    impl_unsigned_morton3!(Morton3u8, u32, u8);
    impl_unsigned_morton3!(Morton3u16, u64, u16);
    impl_unsigned_morton3!(Morton3u32, u128, u32);
}
pub use impl_unsigned::*;

mod impl_signed {
    use morton_encoding::{morton_decode, morton_encode};

    #[cfg(feature = "rkyv")]
    use rkyv::{Archive, Deserialize, Serialize};

    macro_rules! impl_signed_morton2 {
        ($morton:ident, $store:ident, $scalar:ident, $translate_fn:ident, $untranslate_fn:ident) => {
            #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
            #[cfg_attr(
                feature = "rkyv",
                derive(Archive, Deserialize, Serialize),
                archive_attr(derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord))
            )]
            pub struct $morton(pub $store);

            impl From<$morton> for [$scalar; 2] {
                #[inline]
                fn from(m: $morton) -> Self {
                    let [y, x] = morton_decode(m.0);
                    [$untranslate_fn(x), $untranslate_fn(y)]
                }
            }
            impl From<[$scalar; 2]> for $morton {
                #[inline]
                fn from([x, y]: [$scalar; 2]) -> $morton {
                    $morton(morton_encode([$translate_fn(y), $translate_fn(x)]))
                }
            }
        };
    }

    macro_rules! impl_signed_morton3 {
        ($morton:ident, $store:ident, $scalar:ident, $translate_fn:ident, $untranslate_fn:ident) => {
            #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
            #[cfg_attr(
                feature = "rkyv",
                derive(Archive, Deserialize, Serialize),
                archive_attr(derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord))
            )]
            pub struct $morton(pub $store);

            impl From<$morton> for [$scalar; 3] {
                #[inline]
                fn from(m: $morton) -> Self {
                    let [z, y, x] = morton_decode(m.0);
                    [$untranslate_fn(x), $untranslate_fn(y), $untranslate_fn(z)]
                }
            }
            impl From<[$scalar; 3]> for $morton {
                #[inline]
                fn from([x, y, z]: [$scalar; 3]) -> $morton {
                    $morton(morton_encode([
                        $translate_fn(z),
                        $translate_fn(y),
                        $translate_fn(x),
                    ]))
                }
            }
        };
    }

    impl_signed_morton2!(Morton2i8, u16, i8, translate_i8, untranslate_i8);
    impl_signed_morton2!(Morton2i16, u32, i16, translate_i16, untranslate_i16);
    impl_signed_morton2!(Morton2i32, u64, i32, translate_i32, untranslate_i32);

    impl_signed_morton3!(Morton3i8, u32, i8, translate_i8, untranslate_i8);
    impl_signed_morton3!(Morton3i16, u64, i16, translate_i16, untranslate_i16);
    impl_signed_morton3!(Morton3i32, u128, i32, translate_i32, untranslate_i32);

    fn translate_i8(x: i8) -> u8 {
        x.wrapping_sub(i8::MIN) as u8
    }

    fn translate_i16(x: i16) -> u16 {
        x.wrapping_sub(i16::MIN) as u16
    }

    fn translate_i32(x: i32) -> u32 {
        x.wrapping_sub(i32::MIN) as u32
    }

    fn untranslate_i8(x: u8) -> i8 {
        (x as i8).wrapping_add(i8::MIN)
    }

    fn untranslate_i16(x: u16) -> i16 {
        (x as i16).wrapping_add(i16::MIN)
    }

    fn untranslate_i32(x: u32) -> i32 {
        (x as i32).wrapping_add(i32::MIN)
    }
}
pub use impl_signed::*;
