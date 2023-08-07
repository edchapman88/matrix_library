use std::error::Error;
use std::fmt::Display;
use std::ops::{Add, Mul};

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

impl Matrix<usize> {
    pub fn transpose(&self) -> Matrix<usize> {
        let mut res = vec![vec![0_usize; self.nrows]; self.ncols];
        for i in 0..self.ncols {
            for j in 0..self.nrows {
                res[i][j] = self.values[j][i];
            }
        }
        Matrix::new(res)
    }
    pub fn matmul(&self, b: &Matrix<usize>) -> Result<Matrix<usize>, MatrixError> {
        if self.ncols == b.nrows {
            let mut res = vec![vec![0_usize; b.ncols]; self.nrows];
            for j_a in 0..self.nrows {
                for i_b in 0..b.ncols {
                    let mut dot_prod = 0_usize;
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

impl Add for Matrix<usize> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut res = vec![vec![0_usize; self.ncols]; self.nrows];
        for j in 0..self.nrows {
            for i in 0..self.ncols {
                res[j][i] = self.values[j][i] + rhs.values[j][i];
            }
        }
        Matrix::new(res)
    }
}

impl Mul for Matrix<usize> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = vec![vec![0_usize; self.ncols]; self.nrows];
        for j in 0..self.nrows {
            for i in 0..self.ncols {
                res[j][i] = self.values[j][i] * rhs.values[j][i];
            }
        }
        Matrix::new(res)
    }
}

impl Matrix<f64> {
    pub fn transpose(&self) -> Matrix<f64> {
        let mut res = vec![vec![0.0; self.nrows]; self.ncols];
        for i in 0..self.ncols {
            for j in 0..self.nrows {
                res[i][j] = self.values[j][i];
            }
        }
        Matrix::new(res)
    }
    pub fn matmul(&self, b: &Matrix<f64>) -> Result<Matrix<f64>, MatrixError> {
        if self.ncols == b.nrows {
            let mut res = vec![vec![0.0; b.ncols]; self.nrows];
            for j_a in 0..self.nrows {
                for i_b in 0..b.ncols {
                    let mut dot_prod = 0.0;
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

impl Add for Matrix<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut res = vec![vec![0.0; self.ncols]; self.nrows];
        for j in 0..self.nrows {
            for i in 0..self.ncols {
                res[j][i] = self.values[j][i] + rhs.values[j][i];
            }
        }
        Matrix::new(res)
    }
}

impl Mul for Matrix<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = vec![vec![0.0; self.ncols]; self.nrows];
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
    use super::*;
    use std::time::Instant;
    use std::vec;

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

    #[test]
    fn benchmark_matmul() {
        let n_itr = 100;

        fn time_matmul(a: Matrix<f64>, b: Matrix<f64>, n_itr: u32) -> (f64, f64) {
            let mut times = Vec::new();
            for _ in 0..n_itr {
                let now = Instant::now();
                {
                    a.matmul(&b);
                }
                let elapsed = now.elapsed();
                times.push(elapsed.as_secs_f64() * 1000.0); // s -> ms
            }
            // println!("{:?}",times);
            let times_sqrt: Vec<f64> = times.iter().map(|x| x.powf(2.0)).collect();

            let x_sum: f64 = times.iter().sum();
            let x_sqrt_sum: f64 = times_sqrt.iter().sum();
            let x_bar: f64 = x_sum / f64::from(n_itr);

            let var = (x_sqrt_sum / f64::from(n_itr)) - x_bar.powf(2.0);
            (x_bar, var.sqrt())
        }

        let a = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);

        let (mut x_bar, mut stnd) = time_matmul(a, b, n_itr);
        println!("small matmul: {:.5} ms ({:.5} ms std)", x_bar, stnd);

        let dim_size = 100;
        let a = Matrix::new(vec![vec![5.0; dim_size]; dim_size]);
        let b = Matrix::new(vec![vec![5.0; dim_size]; dim_size]);

        let (mut x_bar, mut stnd) = time_matmul(a, b, n_itr);
        println!(
            "({},{}) matmul: {:.2} ms ({:.2} ms std)",
            dim_size, dim_size, x_bar, stnd
        );
    }
}
