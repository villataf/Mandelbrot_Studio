//! WGSL compute shaders: single precision, emulated double precision and native double precision.

const COMMON: &str = r#"
struct Params {
    width: u32,
    height: u32,
    row_offset: u32,
    max_iter: u32,
    band_rows: u32,
    out_row0: u32,
    use_bulbs: u32,
    // Always 0: see o() in the emulated double precision kernel.
    zero: u32,
    // Each coordinate is split into three f32 parts: hi + mid + lo.
    cx: vec4<f32>,
    cy: vec4<f32>,
    pix: vec4<f32>,
    // x: first column of this dispatch, y: number of columns.
    cols: vec4<u32>,
    // Resumable kernels: x = 1 on the first dispatch, y = iterations per dispatch.
    run: vec4<u32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;

const BAIL: f32 = 65536.0;
const INTERIOR: f32 = -1.0;
const UNKNOWN: f32 = -2.0;

fn smooth_mu(i: u32, mag2: f32) -> f32 {
    let lz = 0.5 * log(mag2);
    return max(f32(i) + 1.0 - log2(lz), 0.0);
}

fn in_main_bulbs(x: f32, y: f32) -> bool {
    let xq = x - 0.25;
    let q = xq * xq + y * y;
    if (q * (q + xq) <= 0.25 * y * y) {
        return true;
    }
    let xb = x + 1.0;
    return xb * xb + y * y <= 0.0625;
}

fn pixel_offsets(gid: vec3<u32>) -> vec2<f32> {
    let row = P.row_offset + gid.y;
    let dx = f32(P.cols.x + gid.x) + 0.5 - 0.5 * f32(P.width);
    let dy = f32(row) + 0.5 - 0.5 * f32(P.height);
    return vec2<f32>(dx, dy);
}

fn store(gid: vec3<u32>, v: f32) {
    OUT[(P.out_row0 + gid.y) * P.width + P.cols.x + gid.x] = v;
}
"#;

const F32_KERNEL: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.cols.y || gid.y >= P.band_rows) {
        return;
    }
    let d = pixel_offsets(gid);
    let cx = P.cx.x + d.x * P.pix.x;
    let cy = P.cy.x - d.y * P.pix.x;
    var result = UNKNOWN;
    if (P.use_bulbs != 0u && in_main_bulbs(cx, cy)) {
        result = INTERIOR;
    } else {
        var x = 0.0;
        var y = 0.0;
        var x2 = 0.0;
        var y2 = 0.0;
        // Cycle detection: an orbit that lands exactly on a saved point repeats forever.
        var sx = 0.0;
        var sy = 0.0;
        var next_save = 8u;
        var i = 0u;
        loop {
            if (i >= P.max_iter) {
                break;
            }
            y = 2.0 * x * y + cy;
            x = x2 - y2 + cx;
            x2 = x * x;
            y2 = y * y;
            i = i + 1u;
            if (x2 + y2 > BAIL) {
                result = smooth_mu(i, x2 + y2);
                break;
            }
            if (x == sx && y == sy) {
                result = INTERIOR;
                break;
            }
            if (i == next_save) {
                sx = x;
                sy = y;
                next_save = next_save * 2u;
            }
        }
    }
    store(gid, result);
}
"#;

/// Double-float arithmetic (a value is hi + lo, two f32). `TWO_PROD_BODY` is replaced
/// by either an FMA-based or a splitting-based exact product.
///
/// Graphics drivers are free to reorder floating-point additions, which would cancel the
/// rounding errors this arithmetic keeps. `o()` passes a value through a bit operation with
/// a uniform the driver cannot see (always 0), so each step is evaluated as written.
const DF_KERNEL: &str = r#"
fn o(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ P.zero);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = o(a + b);
    let bb = o(s - a);
    let e = o(a - o(s - bb)) + o(b - bb);
    return vec2<f32>(s, e);
}

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = o(a + b);
    let e = b - o(s - a);
    return vec2<f32>(s, e);
}

fn split(a: f32) -> vec2<f32> {
    let t = o(4097.0 * a);
    let hi = o(t - o(t - a));
    return vec2<f32>(hi, a - hi);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    TWO_PROD_BODY
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    s.y = s.y + t.x;
    s = quick_two_sum(s.x, s.y);
    s.y = s.y + t.y;
    return quick_two_sum(s.x, s.y);
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var p = two_prod(a.x, b.x);
    p.y = p.y + (a.x * b.y + a.y * b.x);
    return quick_two_sum(p.x, p.y);
}

fn df_mul_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    var p = two_prod(a.x, b);
    p.y = p.y + a.y * b;
    return quick_two_sum(p.x, p.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.cols.y || gid.y >= P.band_rows) {
        return;
    }
    let d = pixel_offsets(gid);
    let pix = vec2<f32>(P.pix.x, P.pix.y);
    let cx = df_add(vec2<f32>(P.cx.x, P.cx.y), df_mul_f(pix, d.x));
    let cy = df_add(vec2<f32>(P.cy.x, P.cy.y), df_mul_f(pix, -d.y));
    var result = UNKNOWN;
    if (P.use_bulbs != 0u && in_main_bulbs(cx.x, cy.x)) {
        result = INTERIOR;
    } else {
        var x = vec2<f32>(0.0, 0.0);
        var y = vec2<f32>(0.0, 0.0);
        var sx = vec2<f32>(0.0, 0.0);
        var sy = vec2<f32>(0.0, 0.0);
        var next_save = 8u;
        var i = 0u;
        loop {
            let x2 = df_mul(x, x);
            let y2 = df_mul(y, y);
            let m = x2.x + y2.x;
            if (m > BAIL) {
                result = smooth_mu(i, m);
                break;
            }
            if (i >= P.max_iter) {
                break;
            }
            let xy = df_mul(x, y);
            y = df_add(vec2<f32>(2.0 * xy.x, 2.0 * xy.y), cy);
            x = df_add(df_add(x2, vec2<f32>(-y2.x, -y2.y)), cx);
            i = i + 1u;
            if (all(x == sx) && all(y == sy)) {
                result = INTERIOR;
                break;
            }
            if (i == next_save) {
                sx = x;
                sy = y;
                next_save = next_save * 2u;
            }
        }
    }
    store(gid, result);
}
"#;

const TWO_PROD_FMA: &str = "let p = o(a * b);\n    return vec2<f32>(p, fma(a, b, -p));";
const TWO_PROD_SPLIT: &str = "let p = o(a * b);\n    let sa = split(a);\n    let sb = split(b);\n    let e = o(o(o(sa.x * sb.x - p) + sa.x * sb.y) + sa.y * sb.x) + sa.y * sb.y;\n    return vec2<f32>(p, e);";

const F64_KERNEL: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.cols.y || gid.y >= P.band_rows) {
        return;
    }
    let d = pixel_offsets(gid);
    let pix = f64(P.pix.x) + f64(P.pix.y) + f64(P.pix.z);
    let cx = (f64(P.cx.x) + f64(P.cx.y) + f64(P.cx.z)) + f64(d.x) * pix;
    let cy = (f64(P.cy.x) + f64(P.cy.y) + f64(P.cy.z)) - f64(d.y) * pix;
    var result = UNKNOWN;
    if (P.use_bulbs != 0u && in_main_bulbs(f32(cx), f32(cy))) {
        result = INTERIOR;
    } else {
        var x = 0.0lf;
        var y = 0.0lf;
        var x2 = 0.0lf;
        var y2 = 0.0lf;
        var sx = 0.0lf;
        var sy = 0.0lf;
        var next_save = 8u;
        var i = 0u;
        loop {
            if (i >= P.max_iter) {
                break;
            }
            y = 2.0lf * x * y + cy;
            x = x2 - y2 + cx;
            x2 = x * x;
            y2 = y * y;
            i = i + 1u;
            let m = f32(x2 + y2);
            if (m > BAIL) {
                result = smooth_mu(i, m);
                break;
            }
            if (x == sx && y == sy) {
                result = INTERIOR;
                break;
            }
            if (i == next_save) {
                sx = x;
                sy = y;
                next_save = next_save * 2u;
            }
        }
    }
    store(gid, result);
}
"#;

/// Perturbation: z = Z + dz, where Z is the reference orbit (computed precisely on the
/// processor) and dz this pixel's small difference from it, followed in single precision.
/// When z comes closer to zero than dz, or the reference runs out, the difference is
/// "rebased" onto the start of the reference, which keeps it small and accurate.
///
/// The whole picture is calculated by every dispatch, but each dispatch only runs up to
/// `P.run.y` more iterations per pixel and saves where each pixel got to in `STATE`; the
/// next dispatch carries on. So all pixels work in parallel (a few slow pixels no longer
/// keep the card waiting piece after piece) while no dispatch lasts long enough for the
/// system to think the card has hung. `PENDING` counts the pixels still unfinished.
const PERTURBATION_KERNEL: &str = r#"
struct State {
    dz: vec2<f32>,
    sdz: vec2<f32>,
    sz: vec2<f32>,
    m: u32,
    n: u32,
    next_save: u32,
    done: u32,
};

@group(0) @binding(2) var<storage, read> REF: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> STATE: array<State>;
@group(0) @binding(4) var<storage, read_write> PENDING: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.cols.y || gid.y >= P.band_rows) {
        return;
    }
    let idx = (P.out_row0 + gid.y) * P.width + P.cols.x + gid.x;
    var st = State(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0e30, 1.0e30), 0u, 0u, 8u, 0u);
    if (P.run.x == 0u) {
        st = STATE[idx];
        if (st.done != 0u) {
            return;
        }
    }
    let d = pixel_offsets(gid);
    // Distance from the reference point; both offsets are whole pixels plus a half.
    let dc = vec2<f32>((d.x - P.cx.w) * P.pix.x, -(d.y - P.cy.w) * P.pix.x);
    let last = P.cols.z - 1u;
    var dz = st.dz;
    var m = st.m;
    var n = st.n;
    var result = UNKNOWN;
    var finished = false;
    // Cycle detection on the whole state (reference value and difference, both exact).
    var sdz = st.sdz;
    var sz = st.sz;
    var next_save = st.next_save;
    let limit = min(P.max_iter, n + P.run.y);
    loop {
        if (n >= limit) {
            break;
        }
        let zr = REF[m];
        let nx = 2.0 * (zr.x * dz.x - zr.y * dz.y) + (dz.x * dz.x - dz.y * dz.y) + dc.x;
        let ny = 2.0 * (zr.x * dz.y + zr.y * dz.x) + 2.0 * dz.x * dz.y + dc.y;
        dz = vec2<f32>(nx, ny);
        m = m + 1u;
        n = n + 1u;
        let z = REF[m] + dz;
        let mz = dot(z, z);
        if (mz > BAIL) {
            result = smooth_mu(n, mz);
            finished = true;
            break;
        }
        if (mz < dot(dz, dz) || m >= last) {
            dz = z;
            m = 0u;
        }
        let zc = REF[m];
        if (all(dz == sdz) && all(zc == sz)) {
            result = INTERIOR;
            finished = true;
            break;
        }
        if (n == next_save) {
            sdz = dz;
            sz = zc;
            next_save = next_save * 2u;
        }
    }
    if (n >= P.max_iter) {
        finished = true;
    }
    if (finished) {
        OUT[idx] = result;
        st.done = 1u;
    } else {
        atomicAdd(&PENDING, 1u);
    }
    st.dz = dz;
    st.sdz = sdz;
    st.sz = sz;
    st.m = m;
    st.n = n;
    st.next_save = next_save;
    STATE[idx] = st;
}
"#;

pub fn perturbation_source() -> String {
    format!("{COMMON}{PERTURBATION_KERNEL}")
}

pub fn f32_source() -> String {
    format!("{COMMON}{F32_KERNEL}")
}

pub fn df_source(use_fma: bool) -> String {
    let body = if use_fma {
        TWO_PROD_FMA
    } else {
        TWO_PROD_SPLIT
    };
    format!("{COMMON}{}", DF_KERNEL.replace("TWO_PROD_BODY", body))
}

pub fn f64_source() -> String {
    format!("{COMMON}{F64_KERNEL}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str, caps: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .expect("shader validates");
    }

    #[test]
    fn shaders_compile() {
        validate(&f32_source(), naga::valid::Capabilities::empty());
        validate(&perturbation_source(), naga::valid::Capabilities::empty());
        validate(&df_source(true), naga::valid::Capabilities::empty());
        validate(&df_source(false), naga::valid::Capabilities::empty());
        validate(&f64_source(), naga::valid::Capabilities::FLOAT64);
    }
}
