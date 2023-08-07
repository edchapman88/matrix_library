pub trait Element {
    fn zero() -> Self;
}

impl Element for usize {
    fn zero() -> Self {
        0
    }
}

impl Element for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl Element for u8 {
    fn zero() -> Self {
        0
    }
}

impl Element for u32 {
    fn zero() -> Self {
        0
    }
}

impl Element for i32 {
    fn zero() -> Self {
        0
    }
}
