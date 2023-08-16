pub trait Exp {
    fn exp(self) -> Self;
}

pub trait Pow<T = Self> {
    fn pow(self, exp: T) -> Self;
}
