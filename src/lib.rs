

// trait MatrixLike<const J: usize, const I: usize> {
//     fn shape(&self) -> (usize,usize);

//     fn transpose(&self) -> dyn MatrixLike<I,J>; // how precise can this be at the trait level?

//     fn dot(&self, other: Self) -> usize;

//     fn elemwise_add(&self, other: Self) -> Self;
//     fn elemwise_product(&self, other: Self) -> Self;

//     fn add<const K: usize>(&self, other: impl MatrixLike<J, K>) -> Box<dyn MatrixLike<J,I+K>>; //how to achieve this?
//     fn mul(&self, other: Self) -> Self;
// }


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

impl<const J: usize, const I: usize> Matrix<J,I> {
    pub fn transpose(&self) -> Matrix<I, J> {

        let mut res = [[0_usize; J]; I];
        for i in 0..self.ncols {
            for j in 0..self.nrows {
                res[i][j] = self.values[j][i];
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

    pub fn shape(&self) -> (usize,usize) {
        (self.nrows,self.ncols)
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
}
