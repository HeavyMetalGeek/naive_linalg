use std::fmt;
pub use std::ops::{ Add, AddAssign, Sub, SubAssign, 
                    Mul, MulAssign, Div, DivAssign,
                    Index, IndexMut };
pub use std::convert::{TryFrom, TryInto};
use std::str::FromStr;
use csv;

pub trait Scalar: 
    From<i32> + Clone + Copy + AddAssign + Mul<Output=Self> + Into<f64> 
    + FromStr
{}

impl<T> Scalar for T 
where 
    T: From<i32> + Clone + Copy + AddAssign + Mul<Output=T> + Into<f64> 
        + FromStr
{}

#[derive (Debug, Clone)]
pub struct Matrix<T>
{
    data: Vec<Vec<T>>,
}

impl<T> Matrix<T> 
where
    T: Scalar
{
    // pub fn new(cols: usize) -> Self {
    //     let data = vec![Vec::new(); cols];
    //     Self {
    //         data,
    //     }
    // }
    pub fn load(path: &str) -> Self {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .expect("Failed to parse file");
        let mut got_col_size = false;
        let mut ncols = 0;
        let mut mat: Vec<Vec<T>> = Vec::new();
        for record in reader.records() {
            let result = record.expect("Invalid record");
            if !got_col_size { 
                ncols = result.len();
                mat = vec![Vec::new(); ncols];
                got_col_size = true;
            }
            for col in 0..ncols {
                let value = match result[col].parse() {
                    Ok(v) => v,
                    Err(_) => panic!("Could not convert value to T")
                };
                mat[col].push(value);
            }
        }
        Matrix::from_vec(mat)
    }

    pub fn from_vec(vec: Vec<Vec<T>>) -> Self {
        let result = Self {
            data: vec,
        };
        result.col_check();
        result
    }

    pub fn push_col_mut(&mut self, vec: Vec<T>) {
        let len = vec.len();
        let shape = self.shape();
        if len != shape.0 { 
            panic!("Length {} not compatible for matrix with {} rows", 
                len, shape.0);
        };
        self.data.push(vec);
    }

    pub fn push_col(&self, vec: Vec<T>) -> Self {
        let len = vec.len();
        let shape = self.shape();
        if len != shape.0 { 
            panic!("Length {} not compatible for matrix with {} rows", 
                len, shape.0);
        };
        let mut result = self.clone();
        result.data.push(vec);
        result
    }

    pub fn push_row_mut(&mut self, vec: Vec<T>) {
        let len = vec.len();
        let shape = self.shape();
        if len != shape.1 { 
            panic!("Length {} not compatible for matrix with {} columns", 
                len, shape.1);
        };
        for col in 0..shape.1 {
            self.data[col].push(vec[col]);
        }
    }

    pub fn push_row(&self, vec: Vec<T>) -> Self {
        let len = vec.len();
        let shape = self.shape();
        if len != shape.1 { 
            panic!("Length {} not compatible for matrix with {} columns", 
                len, shape.1);
        };
        let mut result = self.clone();
        for col in 0..shape.1 {
            result.data[col].push(vec[col]);
        }
        result
    }

    fn col_check(&self) {
        let ref_len = self.data[0].len();
        for col in 0..self.ncols() {
            if self.data[col].len() != ref_len {
                panic!("Columns are not equal length!");
            }
        }
    }

    pub fn fill(rows: usize, cols: usize, value: T) -> Self {
        let data = vec![vec![value; rows]; cols];
        Self {
            data,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::fill(rows, cols, 0.into())
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix::fill(rows, cols, 1.into())
    }

    pub fn ncols(&self) -> usize {
        self.data.len()
    }

    pub fn nrows(&self) -> usize {
        self.col_check();
        self.data[0].len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    pub fn dot(&self, rhs: &Matrix<T>) -> Self {
        let a = self.shape();
        let b = rhs.shape();
        if a.1 != b.0 {
            panic!("Incompatible matrix sizes {}x{} and {}x{}", a.0, a.1, b.0, b.1);
        }
        let mut result: Matrix<T> = Matrix::zeros(a.0, b.1);

        for i in 0..a.0 {
            for q in 0..b.1 {
                let mut sum: T = 0.into();
                for (j, p) in (0..a.1).zip(0..b.0) {
                    sum += self.data[j][i] * rhs.data[q][p];
                }
                result.data[q][i] = sum;
            }
        }
        result
    }

    pub fn t(&self) -> Self {
        let shape = self.shape();
        let mut result: Matrix<T> = Matrix::zeros(shape.1, shape.0);
        for row in 0..shape.0 {
            for col in 0..shape.1 {
                result.data[row][col] = self.data[col][row];
            }
        }
        result
    }

    pub fn col_mean(&self) -> Matrix<f64> {
        let shape = self.shape();
        let mut results = vec![Vec::new(); shape.1];
        for col in 0..shape.1 {
            let mut sum: T = 0.into();
            for i in 0..shape.0 {
                sum += self.data[col][i];
            }
            results[col].push(sum.into() / (shape.0 as f64));
        }
        Matrix::from_vec(results)
    }

    pub fn col_std(&self) -> Matrix<f64> {
        let shape = self.shape();
        let means = self.col_mean();
        let mut results = vec![Vec::new(); shape.1];
        for col in 0..shape.1 {
            let mut sum: f64 = 0.into();
            for i in 0..shape.0 {
                let dev = self.data[col][i].into() - means[col][0];
                sum += dev * dev;
            }
            let variance = sum / (shape.0 as f64 - 1.0);
            results[col].push(variance.sqrt());
        }
        Matrix::from_vec(results)
    }

    pub fn normalize(&self) -> Matrix<f64> 
    where
        T: SubAssign
    {
        let shape = self.shape();
        let mu = self.col_mean();
        let sigma = self.col_std();
        let mut result: Matrix<f64> = Matrix::zeros(shape.0, shape.1);
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                if sigma[col][0] == 0.0 {
                    result[col][row] = self[col][row].into();
                } else {
                    let value = self[col][row].into() - mu[col][0];
                    result[col][row] = value / sigma[col][0];
                }
            }
        }
        result
    }
}

/// 
/// Matrix to Matrix arithmatic operator traits
/// 
/// 

impl<T> Add for Matrix<T>
where
    T: Scalar + Add<Output=T>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let shape = self.shape();
        let rshape = rhs.shape();
        if shape != rshape {
            panic!("Incompatable shapes: {}x{} and {}x{}",
                shape.0, shape.1, rshape.0, rshape.1);
        }
        let mut result = Matrix::zeros(shape.0, shape.1);
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                result.data[col][row] = self.data[col][row] + rhs.data[col][row];
            }
        }
        result
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Scalar
{
    fn add_assign(&mut self, rhs: Self) {
        let shape = self.shape();
        let rshape = rhs.shape();
        if shape != rshape {
            panic!("Incompatable shapes: {}x{} and {}x{}",
                shape.0, shape.1, rshape.0, rshape.1);
        }
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                self.data[col][row] += rhs.data[col][row];
            }
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Scalar + Sub<Output=T>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let shape = self.shape();
        let rshape = rhs.shape();
        if shape != rshape {
            panic!("Incompatable shapes: {}x{} and {}x{}",
                shape.0, shape.1, rshape.0, rshape.1);
        }
        let mut result = Matrix::zeros(shape.0, shape.1);
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                result.data[col][row] = self.data[col][row] - rhs.data[col][row];
            }
        }
        result
    }
}

impl<T> SubAssign for Matrix<T>
where
    T: Scalar + SubAssign
{
    fn sub_assign(&mut self, rhs: Self) {
        let shape = self.shape();
        let rshape = rhs.shape();
        if shape != rshape {
            panic!("Incompatable shapes: {}x{} and {}x{}",
                shape.0, shape.1, rshape.0, rshape.1);
        }
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                self.data[col][row] -= rhs.data[col][row];
            }
        }
    }
}

impl<T> Mul for Matrix<T> 
where
    T: Scalar
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let lshape = self.shape();
        let rshape = rhs.shape();
        if lshape != rshape {
            panic!("Matrices must be the same shape: {}x{} and {}x{}", 
                lshape.0, lshape.1, rshape.0, rshape.1);
        }
        let mut result: Matrix<T> = Matrix::zeros(lshape.0, lshape.1);
        for j in 0..lshape.1 {
            for i in 0..lshape.0 {
                result.data[j][i] = self.data[j][i] * rhs.data[j][i];
            }
        }
        result
    }
}

impl<T> MulAssign for Matrix<T> 
where
    T: Scalar + MulAssign
{
    fn mul_assign(&mut self, rhs: Self) {
        let lshape = self.shape();
        let rshape = rhs.shape();
        if lshape != rshape {
            panic!("Matrices must be the same shape: {}x{} and {}x{}", 
                lshape.0, lshape.1, rshape.0, rshape.1);
        }
        for j in 0..lshape.1 {
            for i in 0..lshape.0 {
                self.data[j][i] *= rhs.data[j][i];
            }
        }
    }
}

/// 
/// Scalar to Matrix arithmatic operator traits
/// 
/// 

impl<T> Add<T> for Matrix<T>
where
    T: Scalar + Add<Output=T>
{
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        let shape = self.shape();
        let mut result = Matrix::zeros(shape.0, shape.1);
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                result.data[col][row] = self.data[col][row] + rhs;
            }
        }
        result
    }
}

impl<T> AddAssign<T> for Matrix<T>
where
    T: Scalar
{
    fn add_assign(&mut self, rhs: T) {
        let shape = self.shape();
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                self.data[col][row] += rhs;
            }
        }
    }
}

impl<T> Sub<T> for Matrix<T>
where
    T: Scalar + Sub<Output=T>
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        let shape = self.shape();
        let mut result = Matrix::zeros(shape.0, shape.1);
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                result.data[col][row] = self.data[col][row] - rhs;
            }
        }
        result
    }
}

impl<T> SubAssign<T> for Matrix<T>
where
    T: Scalar + SubAssign
{
    fn sub_assign(&mut self, rhs: T) {
        let shape = self.shape();
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                self.data[col][row] -= rhs;
            }
        }
    }
}

impl<T> Mul<T> for Matrix<T> 
where
    T: Scalar
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        let lshape = self.shape();
        let mut result: Matrix<T> = Matrix::zeros(lshape.0, lshape.1);
        for j in 0..lshape.1 {
            for i in 0..lshape.0 {
                result.data[j][i] = self.data[j][i] * rhs;
            }
        }
        result
    }
}

impl<T> MulAssign<T> for Matrix<T> 
where
    T: Scalar + MulAssign
{
    fn mul_assign(&mut self, rhs: T) {
        let lshape = self.shape();
        for j in 0..lshape.1 {
            for i in 0..lshape.0 {
                self.data[j][i] *= rhs;
            }
        }
    }
}

impl<T> Div<T> for Matrix<T> 
where
    T: Scalar + Div<Output=T>
{
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        let lshape = self.shape();
        let mut result: Matrix<T> = Matrix::zeros(lshape.0, lshape.1);
        for j in 0..lshape.1 {
            for i in 0..lshape.0 {
                result.data[j][i] = self.data[j][i] / rhs;
            }
        }
        result
    }
}

impl<T> DivAssign<T> for Matrix<T> 
where
    T: Scalar + DivAssign
{
    fn div_assign(&mut self, rhs: T) {
        let lshape = self.shape();
        for j in 0..lshape.1 {
            for i in 0..lshape.0 {
                self.data[j][i] /= rhs;
            }
        }
    }
}

impl<T, Idx> Index<Idx> for Matrix<T>
where
    T: Scalar,
    usize: TryFrom<Idx>
{
    type Output = Vec<T>;
    fn index(&self, index: Idx) -> &Vec<T> {
        let i: usize = match index.try_into() {
            Ok(idx) => idx,
            Err(_) => panic!("Index could not be converted to usize")
        };
        &self.data[i]
    }
}

impl<T, Idx> IndexMut<Idx> for Matrix<T>
where
    T: Scalar,
    usize: TryFrom<Idx>
{
    fn index_mut(&mut self, index: Idx) -> &mut Vec<T> {
        let i: usize = match index.try_into() {
            Ok(idx) => idx,
            Err(_) => panic!("Index could not be converted to usize")
        };
        &mut self.data[i]
    }
}

impl<T> fmt::Display for Matrix<T> 
where
    T: Scalar + fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        self.col_check();
        let ncols = self.ncols();
        let nrows = self.nrows();

        let mut out_str = format!("[");
        for row in 0..nrows {
            let spc = if row == 0 {""} else {" "};
            let mut row_str = format!("{}[", spc);
            for col in 0..ncols {
                if col == self.data.len() - 1 {
                    if row == nrows - 1 {
                        row_str = format!("{}{}]", row_str, self.data[col][row]);
                    } else {
                        row_str = format!("{}{}]\n", row_str, self.data[col][row]);
                    }
                } else {
                    row_str = format!("{}{}, ", row_str, self.data[col][row]);
                }
            }
            out_str = format!("{}{}", out_str, row_str);
        }
        write!(f, "{}]", out_str)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
