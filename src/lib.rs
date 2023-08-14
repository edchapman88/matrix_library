use std::collections::VecDeque;
use std::error::Error;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::zip;
use std::ops::AddAssign;
use std::ops::{Add, Mul};

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
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl<T> Matrix<T>
where
    T: Debug + Clone,
{
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
}

impl<T: Clone + Mul<Output = T> + AddAssign> Matrix<T> {
    pub fn matmul(mut self, mut b: Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.ncols != b.nrows {
            return Err(MatrixError::DimMismatch(self.shape(), b.shape()));
        }

        let mut res = VecDeque::new();
        for j_a in 0..self.nrows {
            let mut res_row = VecDeque::new();
            for i_b in 0..b.ncols {
                // prepare column of B for dot product
                let mut b_col = VecDeque::new();
                if j_a < self.nrows - 1 {
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
                    a_row = self.values.get(0).unwrap().clone();
                } else {
                    a_row = self.values.pop_front().unwrap();
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

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

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
        let ans = a.clone().matmul(b);
        assert_eq!(
            ans,
            Ok(Matrix::new(VecDeque::from([
                VecDeque::from([22, 28]),
                VecDeque::from([49, 64]),
            ])))
        );
        assert_eq!(
            a.clone().matmul(a),
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
}
