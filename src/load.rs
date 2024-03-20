use core::iter::zip;
use float_cmp::F64Margin;
use itertools::izip;
use matrix_library::Matrix;
use npyz::NpyFile;
use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io;
use std::time::Instant;

const BENCHMARK_REPEAT: usize = 32768;

// Load an npy file as a Matrix
pub fn load(filename: &str) -> Result<Matrix<f64>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(filename)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;

    read(npy)
}

// Convert an npy input stream into a Matrix
fn read<R: std::io::Read>(npy: NpyFile<R>) -> Result<Matrix<f64>, Box<dyn std::error::Error>> {
    let width = npy.shape()[1];

    let mut coldata = VecDeque::new();
    let mut row = VecDeque::new();
    let mut rowindex = 0;
    for number in npy.data::<f64>()? {
        row.push_back(number?);
        rowindex += 1;
        if rowindex >= width {
            coldata.push_back(row);
            rowindex = 0;
            row = VecDeque::new();
        }
    }

    Ok(Matrix::new(coldata))
}

// Neat approach by budziq from rust-lang.org
// See https://users.rust-lang.org/t/how-to-parse-an-int-from-string/12456/12
fn parse_int(input: &str) -> Option<u32> {
    input
        .chars()
        .skip_while(|ch| !ch.is_digit(10))
        .take_while(|ch| ch.is_digit(10))
        .fold(None, |acc, ch| {
            ch.to_digit(10).map(|b| acc.unwrap_or(0) * 10 + b)
        })
}

// Load in the test data from the npz file
fn load_tests(
    filename: &str,
) -> Result<(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>), Box<dyn std::error::Error>> {
    let mut a = Vec::<Matrix<f64>>::new();
    let mut b = Vec::<Matrix<f64>>::new();
    let mut c = Vec::<Matrix<f64>>::new();

    println!("Reading test matrices from file");

    let file = io::BufReader::new(File::open(filename)?);

    let mut zip = zip::ZipArchive::new(file)?;

    for i in 0..zip.len() {
        let file = zip.by_index(i)?;
        let name = file.name();
        let prefix = file.name().chars().next();
        let num: usize = parse_int(name).unwrap().try_into().unwrap();
        let npy = NpyFile::new(file)?;
        let matrix = read(npy)?;
        match prefix {
            Some('a') => a.insert(num, matrix.clone()),
            Some('b') => b.insert(num, matrix.clone()),
            Some('c') => c.insert(num, matrix.clone()),
            Some(_) => println!("Error"),
            None => println!("Error"),
        }
    }

    Ok((a, b, c))
}

// Run simple unit and benchmarking tests
pub fn run_tests() -> Result<(), Box<dyn Error>> {
    println!("Example matrix manipulation...");
    let a = load("../testdata/matrix-a.npy")?;
    let b = load("../testdata/matrix-b.npy")?;
    let c = load("../testdata/matrix-c.npy")?;

    let d = a.matmul(&b)?;
    let result = c.compare(&d, F64Margin::default());

    println!("Result of A * B:");
    println!("Size: {}, {}", d.shape().0, d.shape().1);
    println!("{}\n", d);

    println!(
        "Matches expected result: {}",
        if result { "Yes" } else { "No" }
    );

    // Perform 512 multiplications and compare against the results from NumPy
    println!("Performing unit tests...");
    let (a, b, c) = load_tests("../testdata/matrices.npz")?;

    let total = a.len();
    let mut passed = 0;
    for (mata, matb, matc) in izip!(&a, &b, &c) {
        let matd = mata.matmul(matb)?;
        if matc.compare(&matd, F64Margin::default()) {
            passed += 1;
        } else {
            println!("Incorrect result");
        }
    }
    println!("Multiplication tests passed: {} out of {}", passed, total);

    println!("Benchmarking...");
    let start_time = Instant::now();
    for _count in 0..BENCHMARK_REPEAT {
        for (mata, matb) in zip(&a, &b) {
            let _matd = mata.matmul(matb)?;
        }
    }
    let elapsed = start_time.elapsed();
    let operations = total * BENCHMARK_REPEAT;
    println!(
        "Time taken to perform {} multiply operations: {:.02} seconds\n",
        operations,
        elapsed.as_millis() as f64 / 1000.0
    );
    let ops_per_sec: f64 = 1000.0 * operations as f64 / elapsed.as_millis() as f64;
    println!("Equivalent to {:.02} operations per second", ops_per_sec);

    Ok(())
}
