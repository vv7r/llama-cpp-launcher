//! Icône de l'application, dessinée par le code : un carré arrondi en
//! dégradé bleu-violet et un triangle « lecture » blanc (un lanceur).
//!
//! Sans dépendance, ce fichier sert deux fois : à l'exécution pour l'icône de
//! la fenêtre (`theme::icon`), et dans `build.rs` (`#[path]`) pour produire
//! le `.ico` intégré à l'exécutable. Les deux sont donc toujours identiques.

/// Pixels RGBA (non prémultipliés) de l'icône en `size` × `size`.
pub fn rgba(size: u32) -> Vec<u8> {
    // Suréchantillonnage 4 × 4 : bords lisses même en 16 × 16.
    const SS: u32 = 4;
    let n = size as usize;
    let mut out = vec![0u8; n * n * 4];
    for y in 0..size {
        for x in 0..size {
            let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for sy in 0..SS {
                for sx in 0..SS {
                    let u = (x as f32 + (sx as f32 + 0.5) / SS as f32) / size as f32;
                    let v = (y as f32 + (sy as f32 + 0.5) / SS as f32) / size as f32;
                    if let Some([pr, pg, pb]) = sample(u, v) {
                        r += pr;
                        g += pg;
                        b += pb;
                        a += 1.0;
                    }
                }
            }
            if a > 0.0 {
                let i = (y as usize * n + x as usize) * 4;
                out[i] = (r / a).round() as u8;
                out[i + 1] = (g / a).round() as u8;
                out[i + 2] = (b / a).round() as u8;
                out[i + 3] = (a / (SS * SS) as f32 * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Couleur au point (u, v) de [0, 1]², ou `None` hors de l'icône.
fn sample(u: f32, v: f32) -> Option<[f32; 3]> {
    // Carré arrondi.
    let radius = 0.22;
    let dx = (radius - u).max(u - (1.0 - radius)).max(0.0);
    let dy = (radius - v).max(v - (1.0 - radius)).max(0.0);
    if dx * dx + dy * dy > radius * radius {
        return None;
    }
    // Triangle « lecture », légèrement décalé à droite pour paraître centré.
    let (a, b, c) = ((0.37, 0.25), (0.37, 0.75), (0.78, 0.5));
    if inside(u, v, a, b, c) {
        return Some([255.0, 255.0, 255.0]);
    }
    // Dégradé en diagonale : bleu (haut gauche) vers violet (bas droite).
    let t = (u + v) / 2.0;
    let from = [0x1E as f32, 0x88 as f32, 0xE5 as f32];
    let to = [0x6A as f32, 0x3F as f32, 0xD8 as f32];
    Some([
        from[0] + (to[0] - from[0]) * t,
        from[1] + (to[1] - from[1]) * t,
        from[2] + (to[2] - from[2]) * t,
    ])
}

fn inside(u: f32, v: f32, a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let side = |p: (f32, f32), q: (f32, f32)| (q.0 - p.0) * (v - p.1) - (q.1 - p.1) * (u - p.0);
    let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// Fichier `.ico` contenant l'icône aux tailles usuelles de Windows, en
/// bitmaps 32 bits (BGRA, canal alpha) : aucun encodeur PNG nécessaire.
#[allow(dead_code)] // utilisé par build.rs
pub fn ico() -> Vec<u8> {
    const SIZES: [u32; 6] = [16, 24, 32, 48, 64, 256];
    let images: Vec<Vec<u8>> = SIZES.iter().map(|&s| dib(s)).collect();
    let mut out = Vec::new();
    // ICONDIR
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(SIZES.len() as u16).to_le_bytes());
    // ICONDIRENTRY × n
    let mut offset = 6 + 16 * SIZES.len() as u32;
    for (&s, img) in SIZES.iter().zip(&images) {
        let dim = if s >= 256 { 0 } else { s as u8 }; // 0 = 256
        out.extend_from_slice(&[dim, dim, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes()); // plans
        out.extend_from_slice(&32u16.to_le_bytes()); // bits par pixel
        out.extend_from_slice(&(img.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += img.len() as u32;
    }
    for img in images {
        out.extend_from_slice(&img);
    }
    out
}

/// Une image de l'`.ico` : BITMAPINFOHEADER, pixels BGRA de bas en haut,
/// puis le masque AND (1 bit par pixel, à zéro : l'alpha fait foi).
fn dib(size: u32) -> Vec<u8> {
    let px = rgba(size);
    let n = size as usize;
    let mask_row = (n.div_ceil(32)) * 4;
    let mut out = Vec::with_capacity(40 + n * n * 4 + mask_row * n);
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(size as i32).to_le_bytes());
    out.extend_from_slice(&(2 * size as i32).to_le_bytes()); // couleur + masque
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 24]); // compression, tailles, palette : zéro
    for y in (0..n).rev() {
        for x in 0..n {
            let i = (y * n + x) * 4;
            out.extend_from_slice(&[px[i + 2], px[i + 1], px[i], px[i + 3]]);
        }
    }
    out.resize(out.len() + mask_row * n, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_transparent_center_white() {
        let s = 64;
        let px = rgba(s);
        assert_eq!(px[3], 0, "coin haut gauche transparent");
        let c = ((s / 2 * s + s / 2) * 4) as usize;
        assert_eq!(&px[c..c + 4], &[255, 255, 255, 255], "triangle blanc au centre");
    }

    #[test]
    fn ico_layout() {
        let ico = ico();
        assert_eq!(&ico[..6], &[0, 0, 1, 0, 6, 0]);
        // La dernière entrée (256) pointe sur une image qui finit le fichier.
        let e = 6 + 16 * 5;
        let len = u32::from_le_bytes(ico[e + 8..e + 12].try_into().unwrap()) as usize;
        let off = u32::from_le_bytes(ico[e + 12..e + 16].try_into().unwrap()) as usize;
        assert_eq!(off + len, ico.len());
        assert_eq!(len, 40 + 256 * 256 * 4 + 32 * 256);
    }
}
