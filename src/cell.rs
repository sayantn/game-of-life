use std::{
    mem,
    ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign},
};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Cell {
    Dead = 0,
    Alive = 1,
}

impl From<bool> for Cell {
    fn from(value: bool) -> Self {
        unsafe { mem::transmute(value) }
    }
}

impl Add for Cell {
    type Output = u8;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self as u8 + rhs as u8
    }
}

impl Add<Cell> for u8 {
    type Output = u8;

    #[inline]
    fn add(self, rhs: Cell) -> Self::Output {
        self + rhs as u8
    }
}

impl Add<u8> for Cell {
    type Output = u8;

    #[inline]
    fn add(self, rhs: u8) -> Self::Output {
        self as u8 + rhs
    }
}

impl BitAnd for Cell {
    type Output = Cell;

    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { mem::transmute(self as u8 & rhs as u8) }
    }
}

impl BitOr for Cell {
    type Output = Cell;

    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { mem::transmute(self as u8 | rhs as u8) }
    }
}

impl BitXor for Cell {
    type Output = Cell;

    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { mem::transmute(self as u8 ^ rhs as u8) }
    }
}

impl AddAssign<Cell> for u8 {
    fn add_assign(&mut self, rhs: Cell) {
        *self += rhs as u8;
    }
}

impl BitOrAssign for Cell {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitAndAssign for Cell {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitXorAssign for Cell {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}
