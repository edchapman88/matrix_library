
trait MatrixLike<const J: usize, const I: usize> {
    // type Other<const A: usize, const B: usize>: MatrixLike<A,B>;
    type Transposed: MatrixLike<I,J>;
    type Concat<const Y: usize>: MatrixLike<J,Y>;
    type Concatenated<const X: usize>: MatrixLike<J,X>;

    fn shape(&self) -> (usize,usize);

    fn transpose(&self) -> Self::Transposed;
    fn concat<const Y: usize, const X: usize>(&self, other: Self::Concat<Y>) -> Self::Concatenated<X>;

    // fn elemwise_add(&self, other: Self) -> Self;
    // fn elemwise_product(&self, other: Self) -> Self;

    // fn dot(&self, other: Self) -> usize;
    
    // fn mul(&self, other: Self) -> Self;
}


#[derive(PartialEq, Debug)]
pub struct Matrix<const J: usize, const I: usize> {
    // The below box-less approach would work, but the matricies really need to go on the heap.
    // Even though the compile-time constants do permit stack allocation, it's not a good
    // use of stack memory, which could run out.
    // values: [[usize; I]; J],

    values: Box<[[usize; I]; J]>,
    nrows: usize,
    ncols: usize
}

impl<const J: usize, const I: usize> MatrixLike<J,I> for Matrix<J,I> {
    type Transposed = Matrix<I,J>;
    type Concat<const Y: usize> = Matrix<J,Y>;
    type Concatenated<const X: usize> = Matrix<J,X>;
    // type Other<const A: usize, const B: usize> = Matrix<A,B>;

    fn shape(&self) -> (usize,usize) {
        (self.nrows,self.ncols)
    }

    fn transpose(&self) -> Matrix<I, J> {

        let mut res = [[0_usize; J]; I];
        for i in 0..self.ncols {
            for j in 0..self.nrows {
                res[i][j] = self.values[j][i];
            }
        }   
        Matrix::new(res)
    }

    fn concat<const Y: usize, const X: usize>(&self, other: Self::Concat<Y>) -> Self::Concatenated<X> {
        let mut res = [[0_usize; X]; J];
        for j in 0..self.nrows {
            for i in 0..self.ncols {
                res[j][i] = self.values[j][i]
            }
        }
        for j in 0..other.nrows {
            for i in 0..other.ncols {
                res[j][i+self.ncols] = other.values[j][i]
            }
        }
        Matrix::new(res)
    }
}


impl<const J: usize, const I: usize> Matrix<J,I> {
    pub fn new(data: [[usize; I]; J]) -> Matrix<J,I> {
        Matrix {
            nrows: data.len(),
            ncols: data[0].len(),
            values: Box::new(data),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matrix() {
        let input = [[1,2],[4,5],[7,8]];
        let mat = Matrix::new(input);
        assert_eq!(mat, Matrix {
            values: Box::new([[1_usize,2],[4,5],[7,8]]),
            nrows: 3,
            ncols: 2
        });
    }

    #[test]
    fn shape_method() {
        let mat = Matrix {
            values: Box::new([[1_usize,2],[4,5],[7,8]]),
            nrows: 3,
            ncols: 2
        };
        assert_eq!(mat.shape(),(3,2));
    }

    #[test]
    fn transpose() {
        let mat = Matrix::new([[1_usize,2],[4,5],[7,8]]);
        let res = mat.transpose();
        assert_eq!(res, Matrix::new([[1_usize,4,7],[2,5,8]]));
    }

    #[test]
    fn concatenation() {
        let a = Matrix::new([[1_usize],[2],[3]]);
        let b = Matrix::new([[1_usize,2],[3,4],[5,6]]);
        let res = a.concat(b);
        assert_eq!(res, Matrix::new([[1_usize,1,2],[2,3,4],[3,5,6]]))
    }
}
