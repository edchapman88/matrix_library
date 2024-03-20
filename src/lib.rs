use float_cmp::ApproxEq;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::zip;
use std::ops::AddAssign;
use std::ops::{Add, Mul};

use math_utils::{Exp, Pow};
pub mod math_utils;

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix<T> {
    values: VecDeque<VecDeque<T>>,
    nrows: usize,
    ncols: usize,
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for j in 0..self.nrows {
            if j != 0 {
                write!(f, " ")?;
            }
            write!(f, "[")?;
            for i in 0..self.ncols {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.values[j][i])?;
            }
            write!(f, "]")?;
            if j != self.nrows - 1 {
                write!(f, "\n")?;
            }
        }
        write!(f, "]")
    }
}

#[derive(Debug, PartialEq)]
pub enum MatrixError {
    DimMismatch((usize, usize), (usize, usize)),
}

impl Error for MatrixError {}

impl Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::DimMismatch(a, b) => {
                write!(
                    f,
                    "Error: Matrix dimension ({},{}) is mismatched with ({},{})",
                    a.0, a.1, b.0, b.1
                )
            }
        }
    }
}

impl<T> Matrix<T> {
    pub fn new(data: VecDeque<VecDeque<T>>) -> Matrix<T> {
        Matrix {
            nrows: data.len(),
            ncols: data[0].len(),
            values: data,
        }
    }

    pub fn from_vecs(data: Vec<Vec<T>>) -> Matrix<T> {
        let nrows = data.len();
        let ncols = data[0].len();
        let mut res = VecDeque::new();
        for row in data {
            let mut r = VecDeque::new();
            for el in row {
                r.push_back(el);
            }
            res.push_back(r);
        }
        Matrix {
            values: res,
            nrows,
            ncols,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn transpose(mut self) -> Matrix<T> {
        let mut res = VecDeque::new();

        for _ in 0..self.ncols {
            let mut row = VecDeque::new();
            for j in 0..self.nrows {
                let el = self.values[j]
                    .pop_front()
                    .expect("known total number of elements in matrix");
                row.push_back(el);
            }
            res.push_back(row);
        }
        Matrix::new(res)
    }

    pub fn at(&self, idxs: (usize, usize)) -> Option<&T> {
        if let Some(row) = self.values.get(idxs.0) {
            if let Some(val) = row.get(idxs.1) {
                return Some(val);
            }
        }
        None
    }

    pub fn at_mut(&mut self, idxs: (usize, usize)) -> Option<&mut T> {
        if let Some(row) = self.values.get_mut(idxs.0) {
            if let Some(val) = row.get_mut(idxs.1) {
                return Some(val);
            }
        }
        None
    }
}

impl<T, M> Matrix<T>
where
    T: Copy + ApproxEq<Margin = M>,
    M: Copy + Default,
{
    pub fn compare(&self, other: &Self, margin: M) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for row in self.values.iter().zip(other.values.iter()) {
            for col in row.0.iter().zip(row.1.iter()) {
                if !col.0.approx_eq(*col.1, margin) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T> Iterator for Matrix<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(row) = self.values.get_mut(0) {
            if let Some(el) = row.pop_front() {
                return Some(el);
            } else {
                // pop empty first row
                self.values.pop_front().unwrap();
                // if there is another row, try and return the first element
                if let Some(next_row) = self.values.get_mut(0) {
                    if let Some(next_el) = next_row.pop_front() {
                        return Some(next_el);
                    }
                }
            }
        }
        None
    }
}

impl<T: Exp + Clone> Exp for Matrix<T> {
    fn exp(mut self) -> Matrix<T> {
        for row in self.values.iter_mut() {
            for el in row.iter_mut() {
                *el = el.clone().exp()
            }
        }
        self
    }
}

impl<T: Pow + Clone> Pow<T> for Matrix<T> {
    fn pow(mut self, exp: T) -> Matrix<T> {
        for row in self.values.iter_mut() {
            for el in row.iter_mut() {
                *el = el.clone().pow(exp.clone())
            }
        }
        self
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn fill(shape: (usize, usize), element: T) -> Matrix<T> {
        let mut res = VecDeque::new();
        for _ in 0..shape.0 {
            let mut row = VecDeque::new();
            for _ in 0..shape.1 {
                row.push_back(element.clone());
            }
            res.push_back(row);
        }
        Matrix::new(res)
    }
}

impl<T: Clone + Mul<Output = T> + AddAssign> Matrix<T> {
    pub fn matmul(&self, b_in: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.ncols != b_in.nrows {
            return Err(MatrixError::DimMismatch(self.shape(), b_in.shape()));
        }
        //
        let mut a = self.clone();
        let mut b = b_in.clone();

        let mut res = VecDeque::new();
        for j_a in 0..a.nrows {
            let mut res_row = VecDeque::new();
            for i_b in 0..b.ncols {
                // prepare column of B for dot product
                let mut b_col = VecDeque::new();
                if j_a < a.nrows - 1 {
                    // columns of B must be cloned, so that they may be used again
                    // on the next row of A
                    for row in b.values.iter() {
                        let el: T = row.get(i_b).unwrap().clone();
                        b_col.push_back(el)
                    }
                } else {
                    // processing the final (or only) row of A, so B values may be moved
                    for row in b.values.iter_mut() {
                        b_col.push_back(row.pop_front().unwrap());
                    }
                }

                // prepare row of A for dot product
                let mut a_row: VecDeque<T>;
                if i_b < b.ncols - 1 {
                    // row of A must be cloned for dot with next column of B
                    // (row of interest is always first row, because previous row is moved into a
                    // dot product before this row is considered)
                    a_row = a.values.get(0).unwrap().clone();
                } else {
                    a_row = a.values.pop_front().unwrap();
                }

                // write dot product to result
                let mut dot = a_row.pop_front().unwrap() * b_col.pop_front().unwrap();
                if !a_row.is_empty() {
                    dot = zip(a_row, b_col).fold(dot, |mut dot_acc, (a, b)| {
                        dot_acc += a * b;
                        dot_acc
                    });
                }
                res_row.push_back(dot);
            }
            res.push_back(res_row);
        }
        Ok(Matrix::new(res))
    }
}

impl<T: Add<Output = T>> Add for Matrix<T> {
    type Output = Matrix<T>;
    fn add(mut self, mut rhs: Self) -> Self::Output {
        let mut res = VecDeque::new();
        for _ in 0..self.nrows {
            for i in 0..self.ncols {
                if i == 0 {
                    res.push_back(VecDeque::new());
                }
                res.back_mut().unwrap().push_back(
                    self.values.front_mut().unwrap().pop_front().unwrap()
                        + rhs.values.front_mut().unwrap().pop_front().unwrap(),
                );
                if i == self.ncols - 1 {
                    self.values.pop_front().unwrap();
                    rhs.values.pop_front().unwrap();
                }
            }
        }
        Matrix::new(res)
    }
}

impl<T: Mul<Output = T>> Mul for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let mut res = VecDeque::new();
        for _ in 0..self.nrows {
            for i in 0..self.ncols {
                if i == 0 {
                    res.push_back(VecDeque::new());
                }
                res.back_mut().unwrap().push_back(
                    self.values.front_mut().unwrap().pop_front().unwrap()
                        * rhs.values.front_mut().unwrap().pop_front().unwrap(),
                );
                if i == self.ncols - 1 {
                    self.values.pop_front().unwrap();
                    rhs.values.pop_front().unwrap();
                }
            }
        }
        Matrix::new(res)
    }
}

impl<T: Clone + Add<Output = T>> Add<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs: T) -> Self::Output {
        let fill = Matrix::fill(self.shape(), rhs);
        self + fill
    }
}

impl<T: Clone + Mul<Output = T>> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let fill = Matrix::fill(self.shape(), rhs);
        self * fill
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn shape() {
        let mat = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        assert_eq!(mat.shape(), (2, 3))
    }

    #[test]
    fn transpose() {
        let mat = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        let mat_t = Matrix::new(VecDeque::from([
            VecDeque::from([1, 4]),
            VecDeque::from([2, 5]),
            VecDeque::from([3, 6]),
        ]));
        assert_eq!(mat.transpose(), mat_t)
    }

    #[test]
    fn add_overflow() {
        let a = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        let b = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        let ans = Matrix::new(VecDeque::from([
            VecDeque::from([2, 4, 6]),
            VecDeque::from([8, 10, 12]),
        ]));
        assert_eq!(a + b, ans);
    }

    #[test]
    fn mul_overflow() {
        let a = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        let b = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        let ans = Matrix::new(VecDeque::from([
            VecDeque::from([1, 4, 9]),
            VecDeque::from([16, 25, 36]),
        ]));
        assert_eq!(a * b, ans);
    }

    #[test]
    fn elemwise_exp() {
        impl Exp for f64 {
            fn exp(self) -> Self {
                self.exp()
            }
        }
        let a = Matrix::new(VecDeque::from([
            VecDeque::from([1.0, -2.0]),
            VecDeque::from([4.0, 5.0]),
        ]));
        let res = a.exp();
        assert_eq!(*res.at((0, 0)).unwrap(), 1.0.exp());
        assert_eq!(*res.at((0, 1)).unwrap(), (-2.0).exp());
        assert_eq!(*res.at((1, 0)).unwrap(), 4.0.exp());
        assert_eq!(*res.at((1, 1)).unwrap(), 5.0.exp());
    }

    #[test]
    fn matmul() {
        let a = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        let b = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2]),
            VecDeque::from([3, 4]),
            VecDeque::from([5, 6]),
        ]));
        let ans = a.clone().matmul(&b);
        assert_eq!(
            ans,
            Ok(Matrix::new(VecDeque::from([
                VecDeque::from([22, 28]),
                VecDeque::from([49, 64]),
            ])))
        );
        assert_eq!(
            a.clone().matmul(&a),
            Err(MatrixError::DimMismatch((2, 3), (2, 3)))
        );
    }

    #[test]
    fn display() {
        let a = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2, 3]),
            VecDeque::from([4, 5, 6]),
        ]));
        assert_eq!(format!("{a}"), "[[1, 2, 3]\n [4, 5, 6]]");
    }

    #[test]
    fn iterate() {
        let a = Matrix::new(VecDeque::from([
            VecDeque::from([1, 2]),
            VecDeque::from([4, 5]),
        ]));
        let mut a_itr = (a).into_iter();
        assert_eq!(a_itr.next().unwrap(), 1);
        assert_eq!(a_itr.next().unwrap(), 2);
        assert_eq!(a_itr.next().unwrap(), 4);
        assert_eq!(a_itr.next().unwrap(), 5);
        assert_eq!(a_itr.next(), None);
    }
}
