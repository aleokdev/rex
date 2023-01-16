pub fn floordiv(a: i32, b: i32) -> i32 {
    (a - if a < 0 { b - 1 } else { 0 }) / b
}
