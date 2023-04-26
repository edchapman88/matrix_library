use matrix_library::Matrix;

fn main() {
    let a = Matrix::new(vec![vec![1,2,3,4,5],
                            vec![4,5,6,7,8],
                            vec![6,4,2,8,5],
                            vec![5,6,6,3,9],
                            vec![9,9,9,4,6],
                            vec![5,4,3,7,7]]);
    println!("\nSee the implimented display style:\n\n {a}");

    let b = Matrix::new(vec![vec![1,2],vec![3,4],vec![5,6]]);
    let c = Matrix::new(vec![vec![1,2,3],vec![4,5,6]]);

    println!("\n{b} \n@\n {c} \n= ");
    match b.matmul(&c) {
        Ok(mat) => {
            println!("{mat}");
        },
        Err(err) => {
            println!("\n\nAnd an implimented display for a custom error varient:\n\n {err}")
        }
    }
}