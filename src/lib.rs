use std::error::Error;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul};

use element::Element;
pub mod element;

#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    values: Vec<Vec<T>>,
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

impl<T: Clone> Matrix<T> {
    pub fn new(data: Vec<Vec<T>>) -> Matrix<T> {
        Matrix {
            nrows: data.len(),
            ncols: data[0].len(),
            values: data,
        }
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    pub fn to_vec(&self) -> Vec<Vec<T>> {
        self.values.clone()
    }
}

impl<T> Matrix<T>
where
    T: Element + Clone + Copy + Add + Add<Output = T> + Mul + Mul<Output = T> + AddAssign,
{
    pub fn transpose(&self) -> Matrix<T> {
        let mut res = vec![vec![T::zero(); self.nrows]; self.ncols];
        for i in 0..self.ncols {
            for j in 0..self.nrows {
                res[i][j] = self.values[j][i];
            }
        }
        Matrix::new(res)
    }
    pub fn matmul(&self, b: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.ncols == b.nrows {
            let mut res = vec![vec![T::zero(); b.ncols]; self.nrows];
            for j_a in 0..self.nrows {
                for i_b in 0..b.ncols {
                    let mut dot_prod = T::zero();
                    for i_a in 0..self.ncols {
                        dot_prod += self.values[j_a][i_a] * b.values[i_a][i_b]
                    }
                    res[j_a][i_b] = dot_prod;
                }
            }
            return Ok(Matrix::new(res));
        }
        Err(MatrixError::DimMismatch(self.shape(), b.shape()))
    }
}

impl<T> Add for Matrix<T>
where
    T: Add + Add<Output = T> + Element + Clone + Copy,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut res = vec![vec![T::zero(); self.ncols]; self.nrows];
        for j in 0..self.nrows {
            for i in 0..self.ncols {
                res[j][i] = self.values[j][i] + rhs.values[j][i];
            }
        }
        Matrix::new(res)
    }
}

impl<T> Mul for Matrix<T>
where
    T: Mul + Mul<Output = T> + Element + Clone + Copy,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = vec![vec![T::zero(); self.ncols]; self.nrows];
        for j in 0..self.nrows {
            for i in 0..self.ncols {
                res[j][i] = self.values[j][i] * rhs.values[j][i];
            }
        }
        Matrix::new(res)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn new_matrix() {
        let mat = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(
            mat,
            Matrix {
                ncols: 3,
                nrows: 2,
                values: vec![vec![1, 2, 3], vec![4, 5, 6]],
            }
        );
    }

    #[test]
    fn shape() {
        let mat = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(mat.shape(), (2, 3))
    }

    #[test]
    fn transpose() {
        let mat = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(
            mat.transpose(),
            Matrix {
                values: vec![vec![1, 4], vec![2, 5], vec![3, 6]],
                nrows: 3,
                ncols: 2
            }
        )
    }

    #[test]
    fn add_overflow() {
        let a = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let b = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(a + b, Matrix::new(vec![vec![2, 4, 6], vec![8, 10, 12]]));
    }

    #[test]
    fn mul_overflow() {
        let a = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let b = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(a * b, Matrix::new(vec![vec![1, 4, 9], vec![16, 25, 36]]));
    }

    #[test]
    fn matmul() {
        let a = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let b = Matrix::new(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);

        assert_eq!(
            a.matmul(&b),
            Ok(Matrix::new(vec![vec![22, 28], vec![49, 64]]))
        );
        assert_eq!(a.matmul(&a), Err(MatrixError::DimMismatch((2, 3), (2, 3))));

        //test f64
        let a = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);

        assert_eq!(
            a.matmul(&b),
            Ok(Matrix::new(vec![vec![22.0, 28.0], vec![49.0, 64.0]]))
        );
        assert_eq!(a.matmul(&a), Err(MatrixError::DimMismatch((2, 3), (2, 3))));
    }

    #[test]
    fn display() {
        let a = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(format!("{a}"), "[[1, 2, 3]\n [4, 5, 6]]");
    }

    #[test]
    fn to_vec() {
        let a = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(a.to_vec(), vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }
}
