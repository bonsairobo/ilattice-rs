//! Generic math on [integer lattices](https://en.wikipedia.org/wiki/Integer_lattice) (regular grids).
//!
//! This crate provides types and traits that make it easy to write generic code with integer lattices, i.e. Z<sup>2</sup> and
//! Z<sup>3</sup>. In particular, we provide implementations of [`IntegerVector`](crate::vector::IntegerVector) for `glam`'s
//! [`IVec2`](glam::IVec2), [`IVec3`](glam::IVec3), [`UVec2`](glam::UVec2) and [`UVec3`](glam::UVec3) types. There are also some
//! traits that apply to vectors with real number scalars as well, and those are implemented for [`Vec2`](glam::Vec2),
//! [`Vec3`](glam::Vec3), and [`Vec3A`](glam::Vec3A).

#![deny(clippy::missing_inline_in_public_items)]

pub mod extent;
pub mod vector;

#[cfg(feature = "morton-encoding")]
pub mod morton;

#[cfg(feature = "glam")]
pub use glam;

pub mod prelude {
    pub use super::extent::*;
    pub use super::vector::*;

    #[cfg(feature = "glam")]
    pub use glam::{IVec2, IVec3, UVec2, UVec3, Vec2, Vec3A};

    #[cfg(feature = "morton-encoding")]
    pub use super::morton::*;
}

mod vector_impls;
